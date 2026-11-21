#!/usr/bin/env python3
"""Regression for common-window accounting, not a GPU performance oracle."""
from pathlib import Path
import copy
import importlib.util
import unittest

module = Path(__file__).resolve().parents[1] / 'scripts' / 'bench_serve_replicas.py'
spec = importlib.util.spec_from_file_location('bench_serve_replicas', module)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def requests():
    rows = []
    for i in range(8):
        events = []
        trace = []
        size = 0
        for step in range(6):
            # Split one licensed round into two text deltas. The measurement
            # must use the SECOND delivery timestamp for its whole count.
            for half in range(2):
                content = 'é' if half else 'a'
                size += len(content.encode('utf-8'))
                events.append({'seconds':step + half * .25, 'content':content, 'reasoning':''})
                trace.append({'content_bytes':size, 'committed_tokens':1+step*24})
        rows.append({'seed':i, 'cancelled':False, 'reasoning':'',
            'started_monotonic':100+i*.1, 'events':events, 'completion_tokens':121,
            'server_record':{'request':{'device':i//4}, 'stream_trace':trace}})
    return rows

class ConversationGrowth(unittest.TestCase):
    def test_actual_answers_are_appended_without_mutating_history(self):
        messages = [[{'role':'user', 'content':'Previous question'}]]
        rows = [{'content':'Actual delivered reply', 'reasoning':'retained thinking'}]
        saved = copy.deepcopy(messages)
        first = bench.followup_workload(messages, rows, 'Next question')
        self.assertEqual(messages, saved)
        self.assertEqual(first[0][1], {'role':'assistant','content':'Actual delivered reply','reasoning_content':'retained thinking'})
        self.assertEqual(first[0][-1]['content'], 'Next question')
        second = bench.followup_workload(first, [{'content':'Second reply'}], 'Third question')
        self.assertEqual(len(second[0]), 5)
        for bad in ([], [{'cancelled':True, 'content':'partial'}], [{'error':'failed'}]):
            with self.assertRaises(ValueError): bench.followup_workload(messages, bad, 'Next')


class CommonWindow(unittest.TestCase):
    def test_common_physical_window_uses_last_delta_and_byte_lengths(self):
        result = bench.common_stream_window(requests())
        self.assertTrue(result['qualified_measurement'])
        self.assertAlmostEqual(result['common_start_monotonic'], 100.95)
        self.assertAlmostEqual(result['common_end_monotonic'], 105.25)
        self.assertAlmostEqual(result['seconds'], 4.3)
        self.assertEqual([row['conservative_tokens'] for row in result['rows']],
                         [116] + [92] * 7)
        self.assertTrue(result['numeric_decode_gate_met'])
        self.assertFalse(result['semantic_quality_certified'])

    def test_disjoint_streams_cannot_be_summed(self):
        rows = requests(); rows[-1]['started_monotonic'] += 20
        result = bench.common_stream_window(rows)
        self.assertFalse(result['qualified_measurement'])

    def test_incomplete_evidence_cannot_pass(self):
        for mutation in ('cancel', 'missing_trace', 'byte_mismatch', 'reasoning', 'nonmonotonic'):
            rows = requests()
            if mutation == 'cancel': rows[0]['cancelled'] = True
            elif mutation == 'missing_trace': rows[0]['server_record'].pop('stream_trace')
            elif mutation == 'byte_mismatch': rows[0]['events'][0]['content'] += 'x'
            elif mutation == 'reasoning': rows[0]['reasoning'] = 'hidden work'
            else: rows[0]['server_record']['stream_trace'][3]['committed_tokens'] = 0
            self.assertFalse(bench.common_stream_window(rows)['qualified_measurement'], mutation)

    def test_one_gpu_is_not_physical_c8(self):
        rows = requests()
        for row in rows: row['server_record']['request']['device'] = 0
        result = bench.common_stream_window(rows)
        self.assertFalse(result['qualified_measurement'])
        self.assertFalse(result['numeric_decode_gate_met'])

class PrimingArrivalSpacing(unittest.TestCase):
    def test_only_priming_is_delayed(self):
        self.assertAlmostEqual(bench.phase_arrival_delay('cold',7,.05), .35)
        for phase in ('warm','followup','repeat_current','refill'):
            self.assertEqual(bench.phase_arrival_delay(phase,7,.05), 0)
        self.assertEqual(bench.phase_arrival_delay('churn',7,.05), 3.5)
        for bad in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): bench.phase_arrival_delay('cold',1,bad)


class SerialCorpus(unittest.TestCase):
    def test_sequential_corpus_cannot_claim_concurrent_throughput(self):
        rows = requests()[:2]
        for i,row in enumerate(rows):
            row['server_record']['request']['device'] = 0
            row['started_monotonic'] = i*1000.0
            row['wall_seconds'] = 100.0
        result = bench.serial_stream_windows(rows)
        self.assertTrue(result['qualified_measurement'])
        self.assertTrue(result['no_request_overlap'])
        self.assertEqual(result['actual_clients'],1)
        self.assertFalse(result['numeric_decode_gate_met'])
        rows[1]['server_record']['request']['device'] = 1
        self.assertFalse(bench.serial_stream_windows(rows)['qualified_measurement'])
        rows[1]['server_record']['request']['device'] = 0
        rows[1]['started_monotonic'] = 1.0
        self.assertFalse(bench.serial_stream_windows(rows)['qualified_measurement'])
        rows[1]['error'] = 'test failure'
        self.assertFalse(bench.serial_stream_windows(rows)['qualified_measurement'])


class ExplicitSingleReplica(unittest.TestCase):
    def test_one_card_never_claims_physical_dual_c8(self):
        for count in (1,2,4,8):
            rows = requests()[:count]
            for row in rows: row['server_record']['request']['device'] = 0
            result = bench.common_stream_window(rows, expected_count=count, replica_count=1)
            self.assertTrue(result['qualified_measurement'])
            self.assertEqual(result['streams_per_gpu'], count)
            self.assertEqual(result['replicas'], 1)
            self.assertFalse(result['numeric_decode_gate_met'])
            self.assertFalse(result['physical_c4_c4'])
            self.assertFalse(bench.common_stream_window(rows, expected_count=count)['qualified_measurement'])
            if count>1:
                rows[-1]['server_record']['request']['device'] = 1
                self.assertFalse(bench.common_stream_window(rows, expected_count=count, replica_count=1)['qualified_measurement'])


class SmallerCohorts(unittest.TestCase):
    def test_two_and_four_clients_remain_distinct_from_the_eight_user_gate(self):
        for indexes in ([0,4], [0,1,4,5]):
            original = requests()
            rows = [original[i] for i in indexes]
            result = bench.common_stream_window(rows, expected_count=len(rows))
            self.assertTrue(result['qualified_measurement'])
            self.assertEqual(result['streams'], len(rows))
            self.assertEqual(result['streams_per_gpu'], len(rows)//2)
            self.assertFalse(result['physical_c4_c4'])
            self.assertFalse(result['numeric_decode_gate_met'])
            self.assertFalse(bench.common_stream_window(rows)['qualified_measurement'])
            for row in rows:
                row['server_record']['request']['device'] = 0
            self.assertFalse(bench.common_stream_window(rows, expected_count=len(rows))['qualified_measurement'])


class BoundedWarmWindow(unittest.TestCase):
    def test_six_stream_window_cannot_claim_the_c8_gate(self):
        original = requests()
        rows = [original[i] for i in [0,1,2,4,5,6]]
        for row in rows:
            row['completion_tokens'] *= 10
            for point in row['server_record']['stream_trace']:
                point['committed_tokens'] *= 10
        result = bench.common_stream_window(rows, expected_count=6,
                                            not_before=101.5, not_after=104.5)
        self.assertTrue(result['qualified_measurement'])
        self.assertEqual(result['streams_per_gpu'], 3)
        self.assertEqual(result['common_start_monotonic'], 101.5)
        self.assertEqual(result['common_end_monotonic'], 104.5)
        self.assertGreater(result['aggregate_tokens_per_second'], 160)
        self.assertFalse(result['physical_c4_c4'])
        self.assertFalse(result['numeric_decode_gate_met'])
        self.assertFalse(bench.common_stream_window(rows)['qualified_measurement'])
        self.assertFalse(bench.common_stream_window(rows, expected_count=6,
            not_before=104.5, not_after=104.6)['qualified_measurement'])


class ColdJoinScenario(unittest.TestCase):
    def test_owner_selection_and_fresh_prefix_do_not_mutate_source(self):
        messages = [[{'role':'user','content':f'Original {i}'}] for i in range(8)]
        messages[7][0]['content'] = [{'type':'image_url','image_url':{'url':'data:image/png;base64,abc'}}]
        original = copy.deepcopy(messages)
        workload, plan = bench.cold_join_workload(messages, requests(), 'test-1')
        self.assertEqual(messages, original)
        self.assertEqual([p['source_index'] for p in plan], [0,1,2,4,5,6,3,7])
        self.assertEqual([p['cold'] for p in plan], [False]*6+[True]*2)
        self.assertEqual([p['delay_seconds'] for p in plan], [0]*6+[2]*2)
        self.assertEqual(workload[0], messages[0])
        self.assertTrue(workload[6][0]['content'].endswith('Original 3'))
        self.assertEqual(workload[7][0]['content'][1:], original[7][0]['content'])
        self.assertIn('test-1', workload[7][0]['content'][0]['text'])

    def test_fresh_media_replacements_preserve_warm_owners_and_original_inputs(self):
        messages = [[{'role':'user','content':f'Original {i}'}] for i in range(8)]
        fresh = [[{'role':'user','content':[{'type':'video_url','video_url':{'url':'data:video/mp4;base64,abc'}}]}],
                 [{'role':'user','content':'A fresh image context'}]]
        saved = copy.deepcopy(fresh)
        workload, plan = bench.cold_join_workload(messages, requests(), 'media', fresh)
        self.assertEqual(fresh, saved)
        self.assertEqual(workload[:6], [messages[i] for i in (0,1,2,4,5,6)])
        self.assertEqual(workload[6][0]['content'][1], fresh[0][0]['content'][0])
        self.assertTrue(workload[7][0]['content'].endswith('A fresh image context'))
        self.assertEqual([p['cold'] for p in plan], [False]*6+[True]*2)
        with self.assertRaises(ValueError):
            bench.cold_join_workload(messages, requests(), 'invalid', fresh[:1])

    def test_one_gpu_or_cancelled_history_cannot_qualify(self):
        messages = [[{'role':'user','content':'test'}] for _ in range(8)]
        rows = requests()
        for row in rows: row['server_record']['request']['device'] = 0
        with self.assertRaises(ValueError): bench.cold_join_workload(messages, rows, 'bad')
        rows = requests(); rows[0]['cancelled'] = True
        with self.assertRaises(ValueError): bench.cold_join_workload(messages, rows, 'bad')

    def test_actual_cold_arrival_and_prefix_miss_are_required(self):
        messages = [[{'role':'user','content':'test'}] for _ in range(8)]
        _, plan = bench.cold_join_workload(messages, requests(), 'test')
        rows = []
        for i, spec in enumerate(plan):
            cold = spec['cold']
            # Fresh requests may be routed to either available GPU.
            device = 1-spec['expected_device'] if cold else spec['expected_device']
            rows.append({'seed':i,'cancelled':False,'started_monotonic':102 if cold else 100,
                'ttft_seconds':1 if cold else .1,'last_content_seconds':10,
                'visible_decode_tps_conservative':20,'inter_chunk_max_seconds':.2,
                'server_record':{'request':{'device':device}, 'result':{
                    'prompt_tokens':2000,'computed_prefill_tokens':2000 if cold else 4,
                    'prefix_cache_hit_tokens':0 if cold else 1996}}})
        self.assertTrue(bench.summarize_cold_join(rows, plan)['qualified_scenario'])
        rows[6]['server_record']['result']['computed_prefill_tokens'] = 4
        self.assertFalse(bench.summarize_cold_join(rows, plan)['qualified_scenario'])
        rows[6]['server_record']['result']['computed_prefill_tokens'] = 2000
        rows[0]['last_content_seconds'] = 1
        self.assertFalse(bench.summarize_cold_join(rows, plan)['qualified_scenario'])
        self.assertFalse(bench.summarize_cold_join([], [])['qualified_scenario'])


class TwoColdPerGpu(unittest.TestCase):
    def test_balanced_four_established_four_cold_owner_plan(self):
        messages=[[{'role':'user','content':f'Original {i}'}] for i in range(8)]
        fresh=[[{'role':'user','content':f'Fresh {i}'}] for i in range(4)]
        workload,plan=bench.cold_join_workload(messages,requests(),'two-new',fresh,2)
        self.assertEqual([p['source_index'] for p in plan],[0,1,4,5,2,3,6,7])
        self.assertEqual([p['cold'] for p in plan],[False]*4+[True]*4)
        self.assertEqual([p['delay_seconds'] for p in plan],[0]*4+[2]*4)
        rows=[]
        for i,p in enumerate(plan):
            cold=p['cold'];rows.append({'seed':i,'cancelled':False,'started_monotonic':102 if cold else 100,
                'ttft_seconds':1 if cold else .1,'last_content_seconds':10,
                'visible_decode_tps_conservative':20,'inter_chunk_max_seconds':.2,
                'server_record':{'request':{'device':p['expected_device']},'result':{
                    'prompt_tokens':2000,'computed_prefill_tokens':2000 if cold else 4,
                    'prefix_cache_hit_tokens':0 if cold else 1996}}})
        result=bench.summarize_cold_join(rows,plan)
        self.assertTrue(result['qualified_scenario'])
        self.assertEqual(result['established_streams_per_gpu'],2)
        self.assertIsNone(result['three_warm_streams_active_at_each_join'])
        rows[-1]['server_record']['request']['device']=0
        self.assertFalse(bench.summarize_cold_join(rows,plan)['qualified_scenario'])
        with self.assertRaises(ValueError):bench.cold_join_workload(messages,requests(),'bad',fresh[:2],2)


if __name__ == '__main__':
    unittest.main()

// Executes both packet representations through the same real TP2 shard kernels.
// The old width5-strided packet is an independent control for T4 compaction.
#include "../apps/dual/tp2_mlp.cu"
#include <array>
#include <cstring>
#include <iostream>
namespace ninfer::apps::dual::packet_test {
int run(const std::filesystem::path& root){
    std::uint64_t compared=0;
    for(int device=0;device<2;++device) {
        Engine engine;engine.initialize(device,device==0?EngineKind::Fast:EngineKind::Slow,root,0,0);
        const auto kind=engine.kind;
        for(int tokens:std::array{4,5,4}) {
            const auto layout=detail::tp2_input_packet_layout(tokens);
            std::vector<std::int8_t> codes(layout.code_bytes);
            std::vector<__half> scales(layout.scale_bytes/sizeof(__half));
            for(std::size_t i=0;i<codes.size();++i)codes[i]=static_cast<std::int8_t>((i*29+device*17+tokens*13)%255-127);
            for(std::size_t i=0;i<scales.size();++i)scales[i]=__float2half_rn(float(1+(i%19))/1024.f);
            std::vector<std::uint8_t> oldpacket(kInputPacketBytes,0xa7),packet(layout.packet_bytes,0x5b);
            std::memcpy(oldpacket.data(),codes.data(),codes.size());
            std::memcpy(oldpacket.data()+kMaximumInputLayout.scale_offset,scales.data(),layout.scale_bytes);
            std::memcpy(packet.data(),codes.data(),codes.size());
            std::memcpy(packet.data()+layout.scale_offset,scales.data(),layout.scale_bytes);
            std::vector<std::uint16_t> expected(partial_elements(tokens)),actual(expected.size());
            for(int layer=0;layer<64;++layer) {
                check(cudaSetDevice(device),"test device");
                engine.input_scales=reinterpret_cast<__half*>(engine.input_packet+kMaximumInputLayout.scale_offset);
                check(cudaMemcpyAsync(engine.input_packet,oldpacket.data(),oldpacket.size(),cudaMemcpyHostToDevice,engine.stream),"old packet upload");
                engine.run(layer,nullptr,nullptr,tokens,nullptr);
                check(cudaMemcpyAsync(expected.data(),engine.output_bf16,partial_bf16_bytes(tokens),cudaMemcpyDeviceToHost,engine.stream),"old result");
                engine.synchronize();
                // Poison the whole allocation, so any read past the active packet
                // cannot accidentally match a stale old-format scale/code value.
                check(cudaMemsetAsync(engine.input_packet,0x3d,kInputPacketBytes,engine.stream),"poison reused packet");
                engine.input_scales=reinterpret_cast<__half*>(engine.input_packet+layout.scale_offset);
                check(cudaMemcpyAsync(engine.input_packet,packet.data(),packet.size(),cudaMemcpyHostToDevice,engine.stream),"compact packet upload");
                engine.run(layer,nullptr,nullptr,tokens,nullptr);
                check(cudaMemcpyAsync(actual.data(),engine.output_bf16,partial_bf16_bytes(tokens),cudaMemcpyDeviceToHost,engine.stream),"compact result");
                engine.synchronize();
                if(expected!=actual)throw std::runtime_error("TP2 active-packet parity failed at layer "+std::to_string(layer));
                compared+=expected.size();
            }
            std::cout<<"PASS device="<<device<<" role="<<(kind==EngineKind::Fast?"fast":"slow")
                     <<" tokens="<<tokens<<" layers=64 poisoned_tail exact_BF16_outputs"<<std::endl;
        }
    }
    std::cout<<"TOTAL exact_output_elements="<<compared<<std::endl;
    return 0;
}
}
int main(){try{
    const char* root=std::getenv("NINFER_TP2_PACKET_WEIGHTS");
    if(root==nullptr){std::cout<<"SKIP set NINFER_TP2_PACKET_WEIGHTS for real TP2 source files\n";return 77;}
    return ninfer::apps::dual::packet_test::run(root);
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}

#include "../apps/dual/prefix_snapshot_io.h"
#include "../apps/dual/joined_stage_tasks.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace detail = ninfer::apps::dual::detail;
static_assert(!std::is_copy_constructible_v<detail::PrefixSnapshotBytes>);
static_assert(std::is_nothrow_move_constructible_v<detail::PrefixSnapshotBytes>);

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
struct Directory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ninfer-prefix-read-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Directory() { require(std::filesystem::create_directory(path), "temporary directory collision"); }
    ~Directory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
void test_file_bytes() {
    Directory dir;
    const auto path = dir.path / "prefix.bin";
    std::vector<std::uint8_t> expected(8193);
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = static_cast<std::uint8_t>((i * 157U) ^ (i >> 3));
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(expected.data()), expected.size());
        require(static_cast<bool>(file), "fixture write failed");
    }
    auto bytes = detail::read_prefix_snapshot_bytes(path, expected.size());
    require(bytes.size() == expected.size(), "snapshot size changed");
    require(std::equal(expected.begin(), expected.end(), bytes.data()), "snapshot bytes changed");
    auto moved = std::move(bytes);
    require(std::equal(expected.begin(), expected.end(), moved.data()), "ownership move lost bytes");
    // Both callers have independent storage, including concurrent file reads.
    bool first_ok = false, second_ok = false;
    detail::run_joined_stage_tasks([&] {
        auto a = detail::read_prefix_snapshot_bytes(path, expected.size());
        first_ok = std::equal(expected.begin(), expected.end(), a.data());
    }, [&] {
        auto b = detail::read_prefix_snapshot_bytes(path, expected.size());
        second_ok = std::equal(expected.begin(), expected.end(), b.data());
    });
    require(first_ok && second_ok, "parallel reads changed data");
    for (auto requested : {std::size_t{0}, expected.size()-1, expected.size()+1,
                           std::numeric_limits<std::size_t>::max()}) {
        bool rejected = false;
        try { (void)detail::read_prefix_snapshot_bytes(path, requested); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "invalid snapshot size accepted");
    }
    bool rejected = false;
    try { (void)detail::read_prefix_snapshot_bytes(dir.path / "missing.bin", 8); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "missing snapshot accepted");
    {
        std::ofstream empty(path, std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(empty), "empty fixture creation failed");
    }
    require(detail::read_prefix_snapshot_bytes(path, 0).size() == 0, "empty snapshot read failed");
    rejected = false;
    try { (void)detail::read_prefix_snapshot_bytes(path, expected.size()); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "truncated snapshot accepted");
}
void test_join_and_failure() {
    std::promise<void> started;
    auto event = started.get_future();
    bool first_done = false, second_done = false;
    detail::run_joined_stage_tasks([&] {
        require(event.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "second task did not run concurrently");
        first_done = true;
    }, [&] { started.set_value(); second_done = true; });
    require(first_done && second_done, "task returned before completion");
    for (int failures = 1; failures <= 3; ++failures) {
        std::array<bool, 2> completed{false, false};
        std::string error;
        try {
            detail::run_joined_stage_tasks([&] {
                completed[0] = true;
                if (failures & 1) { throw std::runtime_error("first"); }
            }, [&] {
                completed[1] = true;
                if (failures & 2) { throw std::runtime_error("second"); }
            });
        } catch (const std::exception& failure) { error = failure.what(); }
        require(completed[0] && completed[1], "a failed task left its peer unjoined");
        require(error == ((failures & 1) ? "first" : "second"), "error precedence changed");
    }
    // Callback objects need not be copyable; they remain alive until join.
    bool moved = false;
    detail::run_joined_stage_tasks([] {}, [p = std::make_unique<int>(17), &moved] {
        moved = *p == 17;
    });
    require(moved, "move-only callback failed");
}
}
int main() {
    try {
        test_file_bytes(); test_join_and_failure();
        std::cout << "PASS snapshot bytes/sizes/moves/parallel reads and joined-stage error lifetimes\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

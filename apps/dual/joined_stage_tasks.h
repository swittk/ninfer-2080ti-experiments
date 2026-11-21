#pragma once

#include <exception>
#include <functional>
#include <thread>
#include <utility>

namespace ninfer::apps::dual::detail {

// Both tasks own independent stage state. Finish both before returning or
// propagating errors, so captured buffers/stages cannot outlive this call.
// If both fail, prefer the first-stage error deterministically.
template<class First, class Second>
void run_joined_stage_tasks(First&& first, Second&& second) {
    std::exception_ptr first_error, second_error;
    std::thread worker([&] {
        try { std::invoke(std::forward<Second>(second)); }
        catch (...) { second_error = std::current_exception(); }
    });
    try { std::invoke(std::forward<First>(first)); }
    catch (...) { first_error = std::current_exception(); }
    worker.join();
    if (first_error) { std::rethrow_exception(first_error); }
    if (second_error) { std::rethrow_exception(second_error); }
}

} // namespace ninfer::apps::dual::detail

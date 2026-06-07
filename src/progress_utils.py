from time import perf_counter


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {remaining_seconds:.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"
    return f"{remaining_seconds:.2f}s"


def build_progress_message(label, completed, total, start_time):
    elapsed = perf_counter() - start_time
    if total <= 0 or completed <= 0:
        eta = "unknown"
    else:
        avg_per_unit = elapsed / completed
        eta = format_duration(avg_per_unit * max(total - completed, 0))

    percent = (completed / total * 100) if total > 0 else 0.0
    return (
        f"{label}: {completed}/{total} ({percent:.1f}%) | "
        f"elapsed {format_duration(elapsed)} | ETA {eta}"
    )


def build_weighted_progress_message(label, completed_weight, total_weight, start_time):
    elapsed = perf_counter() - start_time
    if total_weight <= 0 or completed_weight <= 0:
        eta = "unknown"
    else:
        avg_per_weight = elapsed / completed_weight
        eta = format_duration(avg_per_weight * max(total_weight - completed_weight, 0.0))

    percent = (completed_weight / total_weight * 100) if total_weight > 0 else 0.0
    return (
        f"{label}: {percent:.1f}% | "
        f"elapsed {format_duration(elapsed)} | ETA {eta}"
    )

def activity_selection(activities):
    """
    Select the maximum number of non-overlapping activities.

    Args:
        activities (list of tuple): A list of activities where each activity is represented as a tuple (start, end).

    Returns:
        list: A list of selected activities.
    """
    # Sort activities by their finish times
    sorted_activities = sorted(activities, key=lambda x: x[1])

    selected = []
    last_end_time = 0

    for start, end in sorted_activities:
        if start >= last_end_time:
            selected.append((start, end))
            last_end_time = end

    return selected

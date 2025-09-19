from .base import Node

def BeamSearch(
        beam_task,
        depth=10,
        top_k=3,
        num_branch=3,
        max_length=1024,
        initial_branch=None,
        initial_depth=1,
        branch_decay=None,
    ):
    root = Node('')
    cur_nodes = [root]
    orig_branch = num_branch
    current_branch = orig_branch
    valid_output_list = []
    for depth_num in range(depth):
        if depth_num == depth -1:
            stop = "no_stop"
        else:
            stop = None
        if depth_num == 0 and initial_branch is not None:
            current_branch = initial_branch
        candidates = []
        for node in cur_nodes:
            cnt = 3
            if node.is_terminal:
                continue
            new_pcd_list = []
            while not new_pcd_list and cnt:
                # TODO: add termination condition
                new_pcd_list = beam_task.get_next_step(
                    node.y,
                    node.depth + 1,
                    stop=stop,
                    num_branch=current_branch,
                    max_length=max_length,
                    initial_depth=initial_depth,
                )
                cnt -= 1
            if not new_pcd_list:
                continue
            # print(len(new_pcd_list))
            for new_pcd, is_terminal, num_tokens in new_pcd_list:
                print("New pcd: ", new_pcd)
                node, child = node.append_children(new_pcd)
                child.is_terminal = is_terminal
                value = beam_task.get_step_value(child.y)
                child.update_value(value)
                if is_terminal:
                    valid_output_list.append(child)
                beam_task.update_count()
                beam_task.update_budget(num_tokens)
                if not is_terminal:
                    candidates.append(child)

        if not candidates:
            print("No more candidates")
            break
        ranked_candidates = sorted(candidates, key=lambda item: item.V, reverse=True)

        cur_nodes = ranked_candidates[:min(top_k, len(ranked_candidates))]

        if branch_decay is not None:
            current_branch = num_branch // branch_decay
            current_branch = max(orig_branch, num_branch)
            # print(f"Current Branch: {num_branch}")
        else:
            current_branch = orig_branch
        # print(f"Current Branch: {current_branch}")

    # print('If no solution satisfying the required value is found, the highest value value solution is used instead.\n')
    # max_node, max_V = root.getBestV()
    max_node = root.getBestLeaf()
    max_node.final_ans_flag = 1
    return max_node.y, root, max_node
from .base import Node
import random

def DynamicBeamSearch(
        beam_task,
        depth=10,
        beam_size=1,
        target_num_gens=7,
        max_length=1024,
        tau=2
    ):
    root = Node('')
    cur_nodes = [root]
    current_topk = target_num_gens
    # current_branch = int(current_topk * tau)
    valid_output_list = []
    for depth_num in range(depth):
        if depth_num == depth -1:
            stop = "no_stop"
        else:
            stop = None
        candidates = []
        for node in cur_nodes:
            cnt = 3
            if node.is_terminal:
                continue
            new_pcd_list = []
            while not new_pcd_list and cnt:
                if depth_num == 0:
                    current_branch = int(current_topk + 1)
                else:
                    current_branch = 1
                # TODO: add termination condition
                new_pcd_list = beam_task.get_next_step(
                    node.y,
                    node.depth + 1,
                    stop=stop,
                    num_branch=current_branch,
                    max_length=max_length,
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
        if len(valid_output_list) >= target_num_gens:
            break
        
        ranked_candidates = sorted(candidates, key=lambda item: item.V, reverse=True)

        # adjust the number of branches and topk dynamically
        current_topk = target_num_gens - len(valid_output_list)
        current_branch = max(1, int((target_num_gens - len(valid_output_list)) * tau))
        
        cur_nodes = ranked_candidates[:min(current_topk, len(ranked_candidates))]

    # print('If no solution satisfying the required value is found, the highest value value solution is used instead.\n')
    # max_node, max_V = root.getBestV()
    assert len(valid_output_list) >= target_num_gens
    max_node = root.getBestLeaf()
    max_node.final_ans_flag = 1
    return max_node.y, root, max_node, valid_output_list
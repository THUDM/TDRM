import math
import random


class Node(object):
    def __init__(self, pcd: str, parent=None, depth=0):
        self.pcd = pcd  # current step
        self.children = []
        self.V = 0
        self.parent = parent
        self.y = []  # overall steps
        self.depth = depth
        self.visit_sequence = 0
        self.final_ans_flag = 0
        self.is_terminal = False

    def append_children(self, new_pcd: str):
        node = Node(new_pcd, self, self.depth + 1)
        node.update_y_from_parent()
        self.children.append(node)
        return self, node

    def update_y_from_parent(self):
        if self.parent is None:
            self.y = self.pcd
        else:
            # print("self.pcd: {}, self.parent.y: {}".format(self.pcd, self.parent.y))
            self.y = self.parent.y + list(self.pcd)

    def update_value(self, value):
        self.V = value

    def getBestV(self):  # Gets the subtree maximum value node
        if not self.children:
            return self, self.V
        max_V = self.V
        max_node = self
        for child in self.children:
            subNode, subValue = child.getBestV()
            if subValue >= max_V:
                max_V = subValue
                max_node = subNode
        
        return max_node, max_V
    
    def getBestLeaf(self):
        if not self.children:
            return self
        max_node = self
        if not max_node.is_terminal:
            max_node.V = -math.inf
        for child in self.children:
            subNode = child.getBestLeaf()
            if subNode.V >= max_node.V and subNode.is_terminal:
                max_node = subNode
        return max_node

    def get_multiply_value(self):
        if self.depth == 0:
            return 0
        multi_value = self.V
        cur_node = self.parent
        while cur_node.depth > 0:
            multi_value = multi_value * cur_node.V
            cur_node = cur_node.parent
        return multi_value

    def traverse(self, node_list, id, pre_id):
        node_list.append((self.pcd, float(self.V), self.y, self.depth, id, pre_id, self.is_terminal))
        for child in self.children:
            child.traverse(node_list, len(node_list), id)

        return node_list
    
class LookaheadNode(Node):
    def __init__(self, pcd: str, parent=None, depth=0):
        super().__init__(pcd, parent, depth)
        self.rollout = ""
        self.rollout_V = 0

    def getBestV(self):
        if not self.children:
            return self, self.V
        max_V = self.V
        max_node = self
        for child in self.children:
            subNode, subValue = child.getBestV()
            if subValue >= max_V:
                max_V = subValue
                max_node = subNode
        
        return max_node, max_V
    
    def append_children(self, new_pcd: str):
        node = LookaheadNode(new_pcd, self, self.depth + 1)
        node.update_y_from_parent()
        self.children.append(node)
        return self, node


class SolutionStep(object):
    def __init__(self, x, stp, all_steps, score, step_num):
        self.x = x
        self.stp = stp
        self.all_steps = all_steps
        self.score = score
        self.step_num = step_num


def rand_select(data_list: list, probs: list):  # Sampling by probability
    assert len(data_list) == len(probs), "length do not match!"
    probs_norm = []
    sum_prob = sum(probs)
    for i in probs:
        probs_norm.append(i / sum_prob)
    intervals = []
    count = 0
    for i in probs_norm:
        count = count + i
        intervals.append(count)
    # assert count == 1, "probs error!"
    intervals[len(intervals) - 1] = 1
    index = 0
    rand_prob = random.random()
    while rand_prob >= intervals[index]:
        index = index + 1
    return index, data_list[index]
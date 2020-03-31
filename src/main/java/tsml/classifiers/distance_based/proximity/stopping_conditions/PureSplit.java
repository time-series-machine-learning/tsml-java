package tsml.classifiers.distance_based.proximity.stopping_conditions;

import tsml.classifiers.distance_based.proximity.ProxTree;
import tsml.classifiers.distance_based.proximity.splitting.Split;
import tsml.classifiers.distance_based.utils.tree.TreeNode;

public class PureSplit implements ProxTree.StoppingCondition {
    @Override public boolean shouldStop(final TreeNode<Split> node) {
        final Split split = node.getElement();
        return split.getScore() == 0;
    }
}

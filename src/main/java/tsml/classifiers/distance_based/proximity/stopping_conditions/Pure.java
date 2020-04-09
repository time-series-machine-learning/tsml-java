package tsml.classifiers.distance_based.proximity.stopping_conditions;

import java.util.List;
import tsml.classifiers.distance_based.proximity.ProxTree;
import tsml.classifiers.distance_based.proximity.splitting.Split;
import tsml.classifiers.distance_based.proximity.splitting.scoring.ScoreUtils;
import tsml.classifiers.distance_based.utils.tree.TreeNode;
import utilities.ArrayUtilities;
import utilities.Utilities;
import utilities.class_counts.ClassCounts;

public class PureSplit implements ProxTree.StoppingCondition {
    @Override public boolean shouldStop(final TreeNode<Split> node) {
        return Utilities.isHomogeneous(node.getElement().getData());
    }
}

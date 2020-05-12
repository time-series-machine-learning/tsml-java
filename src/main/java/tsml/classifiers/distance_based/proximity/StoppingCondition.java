package tsml.classifiers.distance_based.proximity;

import java.io.Serializable;
import tsml.classifiers.distance_based.proximity.splitting.Split;
import tsml.classifiers.distance_based.utils.tree.TreeNode;

public interface StoppingCondition extends Serializable {
    boolean shouldStop(TreeNode<Split> node);
}

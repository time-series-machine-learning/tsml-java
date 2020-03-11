package tsml.classifiers.distance_based.proximity;

import java.util.List;
import java.util.ListIterator;

import tsml.classifiers.distance_based.proximity.tree.TreeNode;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ProxTree extends BaseClassifier {

    public ProxTree() {

    }

    private TreeNode<Split> root;
    private ListIterator<TreeNode<Split>> nodeIterator;
    private NodeIteratorBuilder nodeIteratorBuilder = null; // todo
    private SplitBuilder splitBuilder = null; // todo
    private StoppingCondition stoppingCondition = null; // todo

    public interface StoppingCondition {
        boolean shouldStop(TreeNode<Split> node);
        // todo some way to set this as the tree ref
    }

    public interface NodeIteratorBuilder {
        ListIterator<TreeNode<Split>> build();
    }

    public interface SplitBuilder {
        Split build(Instances data);
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        final boolean rebuild = isRebuild();
        super.buildClassifier(trainData);
        if(rebuild) {
            nodeIterator = nodeIteratorBuilder.build();
            final Split split = splitBuilder.build(trainData);
            root = new TreeNode<>(split);
            nodeIterator.add(root);
        }
        while(nodeIterator.hasNext()) {
            final TreeNode<Split> node = nodeIterator.next();
            final Instances data = node.getElement().getData();
            final List<Instances> split = node.getElement().split(data);
            for(Instances childData : split) {
                final Split subSplit = splitBuilder.build(childData);
                final TreeNode<Split> child = new TreeNode<>(subSplit);
                final boolean shouldAdd = stoppingCondition.shouldStop(node);
                if(shouldAdd) {
                    nodeIterator.add(child);
                }
            }
        }
    }

    public NodeIteratorBuilder getNodeIteratorBuilder() {
        return nodeIteratorBuilder;
    }

    public void setNodeIteratorBuilder(
            final NodeIteratorBuilder nodeIteratorBuilder) {
        this.nodeIteratorBuilder = nodeIteratorBuilder;
    }

    public SplitBuilder getSplitBuilder() {
        return splitBuilder;
    }

    public void setSplitBuilder(final SplitBuilder splitBuilder) {
        this.splitBuilder = splitBuilder;
    }

    public StoppingCondition getStoppingCondition() {
        return stoppingCondition;
    }

    public void setStoppingCondition(final StoppingCondition stoppingCondition) {
        this.stoppingCondition = stoppingCondition;
    }
}

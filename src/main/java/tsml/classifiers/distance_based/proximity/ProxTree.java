package tsml.classifiers.distance_based.proximity;

import evaluation.storage.ClassifierResults;
import java.io.Serializable;
import java.util.List;
import java.util.ListIterator;

import tsml.classifiers.distance_based.proximity.Split.Builder;
import tsml.classifiers.distance_based.proximity.splitting.exemplar_based.RandomExemplarPerClassPicker;
import tsml.classifiers.distance_based.proximity.splitting.exemplar_based.RandomExemplarSimilaritySplit;
import tsml.classifiers.distance_based.proximity.stopping_conditions.PureSplit;
import tsml.classifiers.distance_based.proximity.tree.BaseTree;
import tsml.classifiers.distance_based.proximity.tree.BaseTreeNode;
import tsml.classifiers.distance_based.proximity.tree.Tree;
import tsml.classifiers.distance_based.proximity.tree.TreeNode;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import tsml.classifiers.distance_based.utils.iteration.LinearListIterator;
import utilities.ClassifierTools;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ProxTree extends BaseClassifier {

    public static void main(String[] args) throws Exception {
        ProxTree pt = new ProxTree();
        pt.setNodeIteratorBuilder(LinearListIterator::new);
        RandomExemplarSimilaritySplit.Builder builder = new RandomExemplarSimilaritySplit.Builder();
        builder.setExemplarPicker(new RandomExemplarPerClassPicker(pt));
        builder.setUseEarlyAbandon(true);
        builder.setParamSpaces();
        pt.setSplitterBuilder(builder);
        ClassifierResults results = ClassifierTools.trainAndTest("/bench/datasets", "GunPoint", 0, pt);
        System.out.println(results.writeSummaryResultsToString());
    }

    public ProxTree() {

    }

    private Tree<Split> tree;
    private ListIterator<TreeNode<Split>> nodeIterator;
    private NodeIteratorBuilder nodeIteratorBuilder = LinearListIterator::new;
    private Split.Builder splitterBuilder = new RandomExemplarSimilaritySplit.Builder();
    private StoppingCondition stoppingCondition = new PureSplit();

    public interface StoppingCondition extends Serializable {
        boolean shouldStop(TreeNode<Split> node);
        // todo some way to set this as the tree ref
    }

    public interface NodeIteratorBuilder extends Serializable {
        ListIterator<TreeNode<Split>> build();
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        final boolean rebuild = isRebuild();
        super.buildClassifier(trainData);
        if(rebuild) {
            tree = new BaseTree<>();
            nodeIterator = nodeIteratorBuilder.build();
            splitterBuilder.setData(trainData);
            final Split split = splitterBuilder.build();
            BaseTreeNode<Split> root = new BaseTreeNode<>(split);
            tree.setRoot(root);
            nodeIterator.add(root);
        }
        while(nodeIterator.hasNext()) {
            final TreeNode<Split> node = nodeIterator.next();
            final Instances data = node.getElement().getData();
            final List<Instances> split = node.getElement().split(data);
            for(Instances childData : split) {
                splitterBuilder.setData(childData);
                final Split subSplit = splitterBuilder.build();
                final TreeNode<Split> child = new BaseTreeNode<>(subSplit);
                final boolean shouldAdd = stoppingCondition.shouldStop(node);
                if(shouldAdd) {
                    nodeIterator.add(child);
                }
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] distribution = new double[getNumClasses()];

    }

    public NodeIteratorBuilder getNodeIteratorBuilder() {
        return nodeIteratorBuilder;
    }

    public ProxTree setNodeIteratorBuilder(
        NodeIteratorBuilder nodeIteratorBuilder) {
        this.nodeIteratorBuilder = nodeIteratorBuilder;
        return this;
    }

    public Builder getSplitterBuilder() {
        return splitterBuilder;
    }

    public ProxTree setSplitterBuilder(Builder splitterBuilder) {
        this.splitterBuilder = splitterBuilder;
        return this;
    }
}

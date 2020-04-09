package tsml.classifiers.distance_based.proximity;

import com.beust.jcommander.internal.Lists;
import evaluation.storage.ClassifierResults;
import java.io.Serializable;
import java.util.List;
import java.util.ListIterator;

import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.proximity.splitting.AbstractSplitterBuilder;
import tsml.classifiers.distance_based.proximity.splitting.BestOfNSplits;
import tsml.classifiers.distance_based.proximity.splitting.Split;
import tsml.classifiers.distance_based.proximity.splitting.SplitterBuilder;
import tsml.classifiers.distance_based.proximity.splitting.exemplar_based.ContinuousDistanceFunctionConfigs;
import tsml.classifiers.distance_based.proximity.splitting.exemplar_based.RandomDistanceFunctionPicker;
import tsml.classifiers.distance_based.proximity.splitting.exemplar_based.RandomExemplarPerClassPicker;
import tsml.classifiers.distance_based.proximity.splitting.exemplar_based.RandomExemplarProximitySplit;
import tsml.classifiers.distance_based.proximity.splitting.Splitter;
import tsml.classifiers.distance_based.proximity.stopping_conditions.Pure;
import tsml.classifiers.distance_based.utils.tree.BaseTree;
import tsml.classifiers.distance_based.utils.tree.BaseTreeNode;
import tsml.classifiers.distance_based.utils.tree.Tree;
import tsml.classifiers.distance_based.utils.tree.TreeNode;
import tsml.classifiers.distance_based.utils.classifier_building.CompileTimeClassifierBuilderFactory;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import tsml.classifiers.distance_based.utils.iteration.LinearListIterator;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import utilities.ClassifierTools;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ProxTree extends BaseClassifier {

    public static final Factory FACTORY = new Factory();

    public static class Factory extends CompileTimeClassifierBuilderFactory<ProxTree> {
        public final ClassifierBuilder<? extends ProxTree> PROXIMITY_TREE_R1_GINI =
            add(new SuppliedClassifierBuilder<>("PROXIMITY_TREE_R1_GINI", Factory::buildProximityTreeR1Gini));

        public final ClassifierBuilder<? extends ProxTree> PROXIMITY_TREE_R5_GINI =
            add(new SuppliedClassifierBuilder<>("PROXIMITY_TREE_R5_GINI", Factory::buildProximityTreeR5Gini));

        public final ClassifierBuilder<? extends ProxTree> PROXIMITY_TREE_R10_GINI =
            add(new SuppliedClassifierBuilder<>("PROXIMITY_TREE_R10_GINI", Factory::buildProximityTreeR10Gini));

        public static ProxTree buildProximityTreeR1Gini() {
            return setProximityTreeR1GiniConfig(new ProxTree());
        }

        public static ProxTree buildProximityTreeR5Gini() {
            return setProximityTreeR5GiniConfig(new ProxTree());
        }

        public static ProxTree buildProximityTreeR10Gini() {
            return setProximityTreeR10GiniConfig(new ProxTree());
        }


        public static ProxTree setProximityTreeR1GiniConfig(ProxTree pt) {
            pt.setNodeIteratorBuilder(LinearListIterator::new);
            pt.setSplitterBuilder(new AbstractSplitterBuilder() {

                @Override
                public Splitter build() {
                    final Instances trainData = getData();
                    Assert.assertNotNull(trainData);
                    final Random random = pt.getRandom();
                    Assert.assertNotNull(random);
                    final RandomExemplarPerClassPicker exemplarPicker = new RandomExemplarPerClassPicker(random);
                    final List<ParamSpace> paramSpaces = Lists.newArrayList(
                        DistanceMeasureConfigs.buildEdSpace(),
                        DistanceMeasureConfigs.buildFullDtwSpace(),
                        DistanceMeasureConfigs.buildFullDdtwSpace(),
                        ContinuousDistanceFunctionConfigs.buildDtwSpace(trainData),
                        ContinuousDistanceFunctionConfigs.buildDdtwSpace(trainData),
                        ContinuousDistanceFunctionConfigs.buildErpSpace(trainData),
                        ContinuousDistanceFunctionConfigs.buildLcssSpace(trainData),
                        DistanceMeasureConfigs.buildMsmSpace(),
                        ContinuousDistanceFunctionConfigs.buildWdtwSpace(),
                        ContinuousDistanceFunctionConfigs.buildWddtwSpace(),
                        DistanceMeasureConfigs.buildTwedSpace()
                    );
                    final RandomDistanceFunctionPicker distanceFunctionPicker = new RandomDistanceFunctionPicker(
                        random, paramSpaces);
                    return new Splitter() {
                        @Override
                        public Split buildSplit(final Instances data) {
                            RandomExemplarProximitySplit split = new RandomExemplarProximitySplit(
                                random, exemplarPicker,
                                distanceFunctionPicker);
                            split.setData(data);
                            return split;
                        }
                    };
                }
            });
            pt.setStoppingCondition(new Pure());
            return pt;
        }

        public static ProxTree setProximityTreeRXGiniConfig(ProxTree pt, int numSplits) {
            setProximityTreeR1GiniConfig(pt);
            Splitter splitter = pt.getSplitter();
            pt.setSplitter(data -> {
                BestOfNSplits bestOfNSplits = new BestOfNSplits(splitter, pt.getRandom(), numSplits);
                bestOfNSplits.setData(data);
                return bestOfNSplits;
            });
            return pt;
        }

        public static ProxTree setProximityTreeR5GiniConfig(ProxTree pt) {
            return setProximityTreeRXGiniConfig(pt, 5);
        }

        public static ProxTree setProximityTreeR10GiniConfig(ProxTree pt) {
            return setProximityTreeRXGiniConfig(pt, 10);
        }

    }

    public static void main(String[] args) throws Exception {
        ProxTree pt = FACTORY.PROXIMITY_TREE_R5_GINI.build();
        ClassifierResults results = ClassifierTools.trainAndTest("/bench/datasets", "GunPoint", 0, pt);
        System.out.println(results.writeSummaryResultsToString()); // todo interfaces or abst classes for fields in
        // this cls, e.g. node iterator?
    }


    public ProxTree() {
        Factory.setProximityTreeR1GiniConfig(this);
    }

    private Tree<Split> tree;
    private ListIterator<TreeNode<Split>> nodeIterator;
    private NodeIteratorBuilder nodeIteratorBuilder = (NodeIteratorBuilder) () -> {
        throw new UnsupportedOperationException();
    };
    private Splitter splitter;
    private SplitterBuilder splitterBuilder = new AbstractSplitterBuilder() {

        @Override
        public Splitter build() {
            throw new UnsupportedOperationException();
        }
    };
    private StoppingCondition stoppingCondition = (StoppingCondition) node -> {
        throw new UnsupportedOperationException();
    };


    public StoppingCondition getStoppingCondition() {
        return stoppingCondition;
    }

    public ProxTree setStoppingCondition(
        final StoppingCondition stoppingCondition) {
        this.stoppingCondition = stoppingCondition;
        return this;
    }

    public Tree<Split> getTree() {
        return tree;
    }

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
            final SplitterBuilder splitterBuilder = getSplitterBuilder();
            Splitter splitter = getSplitter();
            if(splitterBuilder != null) {
                this.splitterBuilder.setData(trainData);
                this.splitterBuilder.setRandom(getRandom());
                splitter = this.splitterBuilder.build();
                setSplitter(splitter);
            }
            TreeNode<Split> root = buildNode(trainData, null);
            tree.setRoot(root);
        }
        while(nodeIterator.hasNext()) {
            final TreeNode<Split> node = nodeIterator.next();
            final List<Instances> partitions = node.getElement().split();
            System.out.println(node.getElement());
            for(Instances childData : partitions) {
                buildNode(childData, node);
            }
        }
    }

    private TreeNode<Split> buildNode(Instances data, TreeNode<Split> parent) {
        final Split split = splitter.buildSplit(data);
        final TreeNode<Split> node = new BaseTreeNode<>(split);
        node.setParent(parent);
        final boolean stop = stoppingCondition.shouldStop(node);
        if(!stop) {
            nodeIterator.add(node);
        }
        return node;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] distribution = new double[getNumClasses()];
        TreeNode<Split> node = tree.getRoot();
        int index = -1;
        while(!node.isLeaf()) {
            final Split split = node.getElement();
            index = split.getPartitionIndexFor(instance);
            final List<TreeNode<Split>> children = node.getChildren();
            node = children.get(index);
        }
//        if(index < 0) {
//            // todo log warning that we haven't done any tree traversal
//            // todo perhaps rand pick result?
//        } else {
            distribution[index]++;
//        }
        return distribution;
    }

    public NodeIteratorBuilder getNodeIteratorBuilder() {
        return nodeIteratorBuilder;
    }

    public ProxTree setNodeIteratorBuilder(
        NodeIteratorBuilder nodeIteratorBuilder) {
        Assert.assertNotNull(nodeIteratorBuilder);
        this.nodeIteratorBuilder = nodeIteratorBuilder;
        return this;
    }

    public SplitterBuilder getSplitterBuilder() {
        return splitterBuilder;
    }

    public ProxTree setSplitterBuilder(SplitterBuilder splitterBuilder) {
        this.splitterBuilder = splitterBuilder;
        return this;
    }

    private Splitter getSplitter() {
        return splitter;
    }

    private ProxTree setSplitter(Splitter splitter) {
        this.splitter = splitter;
        return this;
    }
}

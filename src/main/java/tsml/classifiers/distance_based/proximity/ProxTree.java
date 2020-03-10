package tsml.classifiers.distance_based.proximity;

import evaluation.storage.ClassifierResults;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import tsml.classifiers.distance_based.pf.Scorer;
import tsml.classifiers.distance_based.proximity.tree.TreeNode;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import tsml.classifiers.distance_based.utils.iteration.LinearListIterator;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ProxTree extends BaseClassifier {

    public ProxTree() {

    }

    private static class Split {
        private Instances data;
        private List<Instances> split;

        public Split(Instances data) {
            this.data = data;
        }

        public Instances getData() {
            return data;
        }

        public List<Instances> split() {
            if(split == null) {
                split = new ArrayList<>(); // todo
            }
            return split;
        }
    }

    private int r = 1;
    private Scorer scorer = Scorer.giniScore;
    private TreeNode<Split> root;
    private ListIterator<TreeNode<Split>> nodeIterator;
    private NodeIteratorBuilder nodeIteratorBuilder = LinearListIterator::new;
    private NodeBuilder nodeBuilder = data -> new TreeNode<>(new Split(data));

    public interface NodeIteratorBuilder {
        ListIterator<TreeNode<Split>> build();
    }

    public interface NodeBuilder {
        TreeNode<Split> build(Instances data);
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        final boolean rebuild = isRebuild();
        super.buildClassifier(trainData);
        if(rebuild) {
            nodeIterator = nodeIteratorBuilder.build();
            root = nodeBuilder.build(trainData);
            nodeIterator.add(root);
        }
        while(nodeIterator.hasNext()) {
            final TreeNode<Split> node = nodeIterator.next();
            final List<Instances> split = node.getElement().split();
            for(Instances childData : split) {
                final TreeNode<Split> child = nodeBuilder.build(childData);
                nodeIterator.add(child);
            }
        }
    }

    // todo why are these implemented here?! should be in base classifier

    @Override
    public String getParameters() {
        return null;
    }

    @Override
    public ClassifierResults getTrainResults() {
        return null;
    }

    @Override
    public boolean getEstimateOwnPerformance() {
        return false;
    }

    @Override
    public void setEstimateOwnPerformance(boolean estimateOwnPerformance) {

    }
}

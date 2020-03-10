package tsml.classifiers.distance_based.proximity;

import evaluation.storage.ClassifierResults;
import java.util.List;
import java.util.ListIterator;
import java.util.function.Supplier;
import tsml.classifiers.distance_based.pf.Scorer;
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

    private int r = 1;
    private Scorer scorer = Scorer.giniScore;
    private ProxNode root;
    private ListIterator<ProxNode> nodeIterator;
    private NodeIteratorBuilder nodeIteratorBuilder = LinearListIterator::new;
    private NodeBuilder nodeBuilder = data -> {
        ProxNode node = new ProxNode();
        node.setInputData(data);
        return node;
    };

    public interface NodeIteratorBuilder {
        ListIterator<ProxNode> build();
    }

    public interface NodeBuilder {
        ProxNode build(Instances data);
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
            final ProxNode node = nodeIterator.next();
            final List<Instances> split = node.split();
            for(Instances childData : split) {
                final ProxNode child = nodeBuilder.build(childData);
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

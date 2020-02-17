package tsml.classifiers.distance_based.pf;

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.pf.relative.ExemplarSplit;
import tsml.classifiers.distance_based.pf.relative.RandomExemplarSplitter;
import utilities.ClassifierTools;
import utilities.serialisation.SerConsumer;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class ProximityTree extends EnhancedAbstractClassifier {
    // todo estimate train const / actually do it

    private Tree<ExemplarSplit> tree;
    private Splitter splitter = data -> {
        throw new UnsupportedOperationException();
    };
    private boolean retrain = true;
    private SerConsumer<Instances> trainSetupFunction = (SerConsumer<Instances>) instances -> {};
    private List<Node<ExemplarSplit>> backlog;

    @Override public void setRetrain(final boolean retrain) {
        this.retrain = retrain;
        super.setRetrain(retrain);
    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        super.buildClassifier(trainData);
        if(retrain) {
            trainSetupFunction.accept(trainData);
            tree = new Tree<>();
            backlog = new ArrayList<>();
            ExemplarSplit exemplarSplit = new ExemplarSplit();
            exemplarSplit.setData(trainData);
            Node<ExemplarSplit> node = new Node<>(exemplarSplit);
            backlog.add(node);
            tree.setRoot(node);
        }
        while (hasNext()) {
            next();
        }
    }

    public boolean hasNext() {
        return !backlog.isEmpty();
    }

    public ProximityTree next() {
        Node<ExemplarSplit> node = backlog.remove(0);
        ExemplarSplit exemplarSplit = node.getElement();
        List<Instances> parts = exemplarSplit.split();
        for(Instances part : parts) {
            ExemplarSplit sub = new ExemplarSplit();
            sub.setData(part);
            Node<ExemplarSplit> child = new Node<>(sub);
            node.getChildren().add(child);
            backlog.add(child);
        }
        return this;
    }

    public Splitter getSplitter() {
        return splitter;
    }

    public void setSplitter(final Splitter splitter) {
        this.splitter = splitter;
    }

    public SerConsumer<Instances> getTrainSetupFunction() {
        return trainSetupFunction;
    }

    public void setTrainSetupFunction(final SerConsumer<Instances> trainSetupFunction) {
        this.trainSetupFunction = trainSetupFunction;
    }

    public static void main(String[] args) throws Exception {
        ProximityTree pt = new ProximityTree();
        int seed = 0;
        pt.setSeed(seed);
        pt.setTrainSetupFunction(trainData -> {
            RandomExemplarSplitter randomExemplarSplitter = new RandomExemplarSplitter();
            randomExemplarSplitter.setSeed(pt.getSeed());
            pt.setSplitter(randomExemplarSplitter);
        });
        ClassifierTools.trainAndTest("/bench/datasets", "GunPoint", 0, pt);
    }
}

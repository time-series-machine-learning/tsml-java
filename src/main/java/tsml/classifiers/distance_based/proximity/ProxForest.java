//package tsml.classifiers.distance_based.proximity;
//
//import java.util.ArrayList;
//import java.util.List;
//import tsml.classifiers.distance_based.utils.classifier_building.CompileTimeClassifierBuilderFactory;
//import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
//import utilities.ArrayUtilities;
//import weka.classifiers.Classifier;
//import weka.core.Instance;
//import weka.core.Instances;
//
///**
// * Purpose: // todo - docs - type the purpose of the code here
// * <p>
// * Contributors: goastler
// */
//public class ProxForest extends BaseClassifier {
//
//    public static final Factory FACTORY = new Factory();
//
//    public static class Factory extends CompileTimeClassifierBuilderFactory<ProxForest> {
//        public final ClassifierBuilder<? extends ProxForest> PROXIMITY_FOREST =
//            add(new SuppliedClassifierBuilder<>("PROXIMITY_FOREST", Factory::buildProximityForest));
//
//        public static ProxForest buildProximityForest() {
//            ProxForest proxForest = new ProxForest();
//            proxForest.setConstituentBuilder(data -> ProxTree.FACTORY.PT_R5_GINI.build());
//            return proxForest;
//        }
//    }
//
//    public interface ConstituentBuilder {
//        Classifier build(Instances data);
//    }
//
//    private List<Classifier> constituents = new ArrayList<>();
//    private ConstituentBuilder constituentBuilder = (Instances trainData) -> {
//        throw new UnsupportedOperationException();
//    };
//    private int numConstituentLimit = 100;
//
//    @Override
//    public void buildClassifier(Instances trainData) throws Exception {
//        final boolean rebuild = isRebuild();
//        super.buildClassifier(trainData);
//        if(rebuild) {
//            constituents = new ArrayList<>();
//        }
//        while(constituents.size() < numConstituentLimit) { // todo different stopping condition
//            Classifier constituent = constituentBuilder.build(trainData);
//            constituents.add(constituent);
//        }
//    }
//
//    @Override
//    public double[] distributionForInstance(Instance instance) throws Exception {
//        double[] distribution = new double[getNumClasses()];
//        for(Classifier constituent : constituents) {
//            // todo fix python pf to match
//            // todo use different voting methods / weighting methods
////            double[] constituentDistribution = constituent.distributionForInstance(instance);
//            double classLabel = constituent.classifyInstance(instance);
//            distribution[(int) classLabel]++;
//        }
//        ArrayUtilities.normalise(distribution);
//        return distribution;
//    }
//
//    public ConstituentBuilder getConstituentBuilder() {
//        return constituentBuilder;
//    }
//
//    public ProxForest setConstituentBuilder(
//        final ConstituentBuilder constituentBuilder) {
//        this.constituentBuilder = constituentBuilder;
//        return this;
//    }
//
//    public int getNumConstituentLimit() {
//        return numConstituentLimit;
//    }
//
//    public ProxForest setNumConstituentLimit(final int numConstituentLimit) {
//        this.numConstituentLimit = numConstituentLimit;
//        return this;
//    }
//}

//package machine_learning.classifiers;
//
//import tsml.classifiers.EnhancedAbstractClassifier;
//import tsml.classifiers.Rebuildable;
//import utilities.Utilities;
//import weka.core.Instance;
//import weka.core.Instances;
//
//import java.util.Map;
//
//public class RandomGuesser extends EnhancedAbstractClassifier implements Rebuildable {
//
//    protected Map<Double, Integer> rawClassDistribution;
//    protected Map<Double, Double> classDistribution;
//
//    public RandomGuesser() {
//        super(false);
//    }
//
//    public RandomGuesser(boolean estimateable) {
//        super(estimateable);
//    }
//
//    @Override public void buildClassifier(final Instances data) throws
//                                                                Exception {
//        rawClassDistribution = Utilities.classDistribution(data);
//        classDistribution = Utilities.normalise(rawClassDistribution);
//    }
//
//    @Override public double[] distributionForInstance(final Instance instance) throws
//                                                                               Exception {
//        double v = rand.nextDouble();
//        int index = 0;
//        do {
//            Double probability = classDistribution.get(index);
//            v -= probability;
//            if(v < 0) {
//                return index;
//            }
//            index++;
//        } while(index < classDistribution.size());
//        throw new IllegalStateException("distribution does not add up to 1");
//    }
//
//    @Override public double classifyInstance(final Instance instance) throws
//                                                                      Exception {
//
//    }
//}

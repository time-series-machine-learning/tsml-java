package tsml.classifiers.distance_based.utils.stats.scoring;

import org.junit.Assert;
import utilities.ArrayUtilities;
import utilities.InstanceTools;
import utilities.Utilities;
import weka.core.Instances;

import java.util.List;

import static utilities.ArrayUtilities.*;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ScoreUtils {

    public static void main(String[] args) {
        int limit = 10;
        for(int i = 0; i <= limit; i++) {
            int[] all = new int[]{0, 0};
            for(int j = 0; j < i; j++) {
                all[0] += 1;
            }
            for(int j = i; j < limit; j++) {
                all[1] += 1;
            }
            double[] a = ArrayUtilities.normalise(all);
            final double v = giniScore(a);
            System.out.println(v);
        }
    }

    /**
     * The gini impurity (0.5 worst, 0 best).
     * @param distribution class counts. This will be normalised therefore sum can be >1 (i.e. class counts can be fed straight in).
     * @return
     */
    public static double giniImpurity(double[] distribution) {
        checkEntropyInputs(distribution);
        ArrayUtilities.normalise(distribution, true);
        double entropy = 1;
        for(double proportion : distribution) {
            double score = Math.pow(proportion, 2);
            entropy -= score;
        }
        return entropy;
    }

    public static double giniImpurity(int[] distribution) {
        return giniImpurity(ArrayUtilities.intToDouble(distribution));
    }

    public static double giniImpurity(Instances data) {
        return giniImpurity(InstanceTools.countClasses(data));
    }

    public static double giniImpurity(List<Instances> data) {
        return entropySum(data, ScoreUtils::giniImpurity);
    }

    /**
     * scales gini impurity between 0 and 1 inclusively (1 being best).
     * @param distribution
     * @return
     */
    public static double giniScore(double[] distribution) {
        // find the worst possible score in order to invert
        final double target = 1 - Math.pow(1d / distribution.length, 2) * distribution.length;
        double entropy = giniImpurity(distribution);
        entropy = target - entropy;
        entropy *= 2; // to put on 0 - 1 scale
        return entropy;
    }

    public static void checkEntropyInputs(double[] distribution) {
        final double sum = ArrayUtilities.sum(distribution);
        for(double v : distribution) {
            if(v < 0) {
                throw new IllegalArgumentException("Distributions cannot contain negative numbers.");
            }
        }
    }

    public static double giniScore(Instances data) {
        return giniScore(InstanceTools.countClasses(data));
    }

    public static double infoScore(Instances data) {
        return infoScore(InstanceTools.countClasses(data));
    }

    /**
     * Info gain entropy of the given distribution (1 is worst, 0 best). Distribution is normalised therefore the class counts can be fed directly to this function.
     * @param distribution
     * @return
     */
    public static double infoEntropy(double[] distribution) {
        checkEntropyInputs(distribution);
        ArrayUtilities.normalise(distribution, true);
        double entropy = 0;
        for(double proportion : distribution) {
            double score = proportion * Utilities.log(proportion, 2);
            entropy += score;
        }
        return -entropy;
    }

    public static double infoEntropy(int[] distribution) {
        return infoEntropy(ArrayUtilities.intToDouble(distribution));
    }

    public static double infoEntropy(Instances data) {
        return infoEntropy(InstanceTools.countClasses(data));
    }

    public static double infoEntropy(List<Instances> data) {
        return entropySum(data, ScoreUtils::infoEntropy);
    }

    /**
     * Inverted info gain entropy to the range 0 - 1 (1 is best, 0 worst).
     * @param distribution
     * @return
     */
    public static double infoScore(double[] distribution) {
        final double v = 1d / distribution.length;
        // calculate the worst possible score in order to invert to 0 - 1 range
        double target = Utilities.log(v, 2) * v * distribution.length;
       return -(target + infoEntropy(distribution));
    }

    public static double infoScore(List<Instances> children) {
        return entropySum(children, ScoreUtils::infoScore);
    }

    public static double entropySum(List<Instances> children, EntropyFunction entropyFunction) {
        double sum = 0;
        for(Instances instances : children) {
            final double[] distribution = InstanceTools.countClasses(instances);
            sum += entropyFunction.entropy(distribution);
        }
        return sum;
    }

    public static double giniScore(List<Instances> children) {
        return entropySum(children, ScoreUtils::giniScore);
    }

    public static double giniGain(Instances parent, List<Instances> children) {
        return gain(parent, children, ScoreUtils::giniScore);
    }

    public static double gain(Instances parent, List<Instances> children, EntropyFunction entropyFunction) {
        final double[][] childrenDistributions = instancesToClassDistributions(children);
        final double[] parentDistributions = InstanceTools.countClasses(parent);
        return gain(parentDistributions, childrenDistributions, entropyFunction);
    }

    private static double[][] instancesToClassDistributions(List<Instances> children) {
        double[][] distributions = new double[children.size()][];
        for(int i = 0; i < distributions.length; i++) {
            distributions[i] = InstanceTools.countClasses(children.get(i));
        }
        return distributions;
    }

    private static void checkGainInputs(double[] parentClassCounts, double[][] childrenClassCounts) {
        for(int i = 0; i < childrenClassCounts.length; i++) {
            Assert.assertEquals(parentClassCounts.length, childrenClassCounts[i].length);
        }
    }

    public static double gain(double[] parentClassCounts, double[][] childrenClassCounts,
            EntropyFunction entropyFunction) {
        checkGainInputs(parentClassCounts, childrenClassCounts);
        final double parentClassCountsSum = ArrayUtilities.sum(parentClassCounts);
        final double parentScore = entropyFunction.entropy(ArrayUtilities.normalise(parentClassCounts));
        double childrenScore = 0;
        for(int i = 0; i < childrenClassCounts.length; i++) {
            final double[] childClassCounts = childrenClassCounts[i];
            // find the proportion of classes in the child node
            final double childSum = ArrayUtilities.sum(childClassCounts);
            // find the entropy at the child
            double entropy = entropyFunction.entropy(ArrayUtilities.normalise(childClassCounts));
            //            entropy = 1 - entropy;
            // weight the entropy by the number of cases at the child node
            entropy *= childSum / parentClassCountsSum;
            // add the entropy to the running childrenScore
            childrenScore += entropy;
        }
        return parentScore - childrenScore;
    }

    public static double infoGain(Instances parent, List<Instances> children) {
        final EntropyFunction doubleFunction = ScoreUtils::infoScore;
        return gain(parent, children, doubleFunction);
    }

    public static double giniScore(int[] classCounts) {
        return giniScore(ArrayUtilities.intToDouble(classCounts));
    }

    public static double infoScore(int[] classCounts) {
        return infoScore(ArrayUtilities.intToDouble(classCounts));
    }

    public static double giniGain(int[] parentClassCounts, int[][] childrenClassCounts) {
        return giniGain(ArrayUtilities.intToDouble(parentClassCounts), ArrayUtilities.intToDouble(childrenClassCounts));
    }

    public static double giniGain(double[] parentClassCounts, double[][] childrenClassCounts) {
        return gain(parentClassCounts, childrenClassCounts, ScoreUtils::giniImpurity);
    }

    public static double infoGain(int[] parentClassCounts, int[][] childrenClassCounts) {
        return infoGain(ArrayUtilities.intToDouble(parentClassCounts), ArrayUtilities.intToDouble(childrenClassCounts));
    }

    public static double infoGain(double[] parentClassCounts, double[][] childrenClassCounts) {
        return gain(parentClassCounts, childrenClassCounts, ScoreUtils::infoEntropy);
    }

    public interface EntropyFunction {
        double entropy(double[] distribution);
    }

    public static double chiSquared(final double[] parentClassCounts, final double[][] childrenClassCounts) {
        checkGainInputs(parentClassCounts, childrenClassCounts);
        final double[] parentDistribution = ArrayUtilities.normalise(copy(parentClassCounts), true);
        double sum = 0;
        for(int i = 0; i < childrenClassCounts.length; i++) {
            final double childSum = sum(childrenClassCounts[i]);
            for(int j = 0; j < parentClassCounts.length; j++) {
                double expected = parentDistribution[j] * childSum;
                double v = Math.pow(childrenClassCounts[i][j] - expected, 2);
                v /= expected;
                sum += v;
            }
        }
        return sum;
    }

    public static double chiSquared(int[] parentClassCounts, int[][] childClassCounts) {
        return chiSquared(ArrayUtilities.intToDouble(parentClassCounts), ArrayUtilities.intToDouble(childClassCounts));
    }

    public static double chiSquared(Instances parent, List<Instances> children) {
        final double[][] childrenDistributions = instancesToClassDistributions(children);
        final double[] parentDistributions = InstanceTools.countClasses(parent);
        return chiSquared(parentDistributions, childrenDistributions);
    }
}

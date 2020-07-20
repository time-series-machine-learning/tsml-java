package tsml.classifiers.distance_based.utils.stats.scoring;

import static utilities.Utilities.convert;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;
import java.util.function.Function;

import tsml.classifiers.distance_based.utils.collections.CollectionUtils;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ScoreUtils {
    public static void main(String[] args) {
        int limit = 10;
        for(int i = 0; i <= limit; i++) {
            Double[] all = new Double[] {0d, 0d};
            for(int j = 0; j < i; j++) {
                all[0] += 1d / limit;
            }
            for(int j = i; j < limit; j++) {
                all[1] += 1d / limit;
            }
            final double v = giniImpurityEntropyFromDistribution(Arrays.asList(all));
            System.out.println(v);
        }
    }

    private static class ClassCount {
        private List<Integer> counts;
        private int sum;
        private List<Double> distribution;

        private ClassCount(final List<Integer> counts) {
            setCounts(counts);
        }

        public List<Integer> getCounts() {
            return counts;
        }

        public int getSum() {
            if(sum < 0) {
                sum = ArrayUtilities.sum(getCounts());
            }
            return sum;
        }

        public List<Double> getDistribution() {
            if(distribution == null) {
                int sum = getSum();
                distribution = Utilities.divide(getCounts(), sum);
            }
            return distribution;
        }

        public ClassCount setCounts(final List<Integer> counts) {
            this.counts = counts;
            setDistribution(null);
            setSum(-1);
            return this;
        }

        private ClassCount setDistribution(final List<Double> distribution) {
            this.distribution = distribution;
            return this;
        }

        private ClassCount setSum(final int sum) {
            this.sum = sum;
            return this;
        }
    }

    public static double infoGainEntropyFromClassCounts(Instances data) {
        return infoGainEntropyFromClassCounts(findClassCounts(data));
    }

    public static double infoGainEntropyFromClassCounts(List<Integer> classCounts) {
        return infoGainEntropyFromClassCounts(new ClassCount(classCounts));
    }

    public static double infoGainEntropyFromDistribution(List<Double> classDistribution) {
        double entropy = 0;
        for(Double proportion : classDistribution) {
            double score = proportion * Utilities.log(proportion, 2);
            entropy += score;
        }
        entropy = 0 - entropy;
        return entropy;
    }

    private static double infoGainEntropyFromClassCounts(ClassCount classCount) {
        return infoGainEntropyFromDistribution(classCount.getDistribution());
    }

    public static double gain(List<Integer> parentClassCounts, List<List<Integer>> childClassCounts,
        Function<ClassCount, Double> entropyFunction) {
        ClassCount parentClassCount = new ClassCount(parentClassCounts);
        int parentClassCountSum = parentClassCount.getSum();
        if(parentClassCountSum <= 0) {
            throw new IllegalArgumentException("parent has empty class count");
        }
        double score = entropyFunction.apply(parentClassCount); // find how pure the parent node is
        score -= entropy(childClassCounts, entropyFunction);
        return score;
    }

    public static double entropy(List<List<Integer>> childClassCounts, Function<ClassCount, Double> entropyFunction) {
        double score = 0;
        final int parentClassCountSum = Utilities.sum(childClassCounts);
        for(List<Integer> classCount : childClassCounts) {
            // find the proportion of classes in the child node
            ClassCount childClassCount = new ClassCount(classCount);
            double proportion = (double) childClassCount.getSum() / parentClassCountSum;
            // find the entropy at the child
            double entropy = entropyFunction.apply(childClassCount);
            // weight the entropy by the number of cases at the child node
            entropy *= proportion;
            // subtract the child entropy from the parent for relative improvement
            score += entropy;
        }
        return score;
    }

    /**
     * larger value -> better gain
     * @param parentClassCount
     * @param childClassCounts
     * @return
     */
    public static double infoGain(List<Integer> parentClassCount, List<List<Integer>> childClassCounts) {
        return gain(parentClassCount, childClassCounts, ScoreUtils::infoGainEntropyFromClassCounts);
    }

    public static double infoGain(Instances parentData, List<Instances> childData) {
        return infoGain(findClassCounts(parentData), findClassCounts(childData));
    }

    public static double infoGainEntropy(List<Instances> childData) {
        return 1 - entropy(findClassCounts(childData), ScoreUtils::infoGainEntropyFromClassCounts); // todo should it
        // be 1- for IG as well??
    }

    public static double giniImpurityEntropyFromClassCounts(Instances data) {
        return giniImpurityEntropyFromClassCounts(findClassCounts(data));
    }

    /**
     * lower value -> more pure
     * @param classCounts
     * @return
     */
    public static double giniImpurityEntropyFromClassCounts(List<Integer> classCounts) {
        return giniImpurityEntropyFromClassCounts(new ClassCount(classCounts));
    }

    public static double giniImpurityEntropyFromClassCounts(ClassCount classCounts) {
        return giniImpurityEntropyFromDistribution(classCounts.getDistribution());
    }

    public static double giniImpurityEntropyFromDistribution(List<Double> distribution) {
        double entropy = 0;
        for(Double proportion : distribution) {
            double score = Math.pow(proportion, 2);
            entropy += score;
        }
        return 1 - entropy;
    }

    /**
     * larger value -> better gain
     * @param parentClassCount
     * @param childClassCounts
     * @return
     */
    public static double giniImpurity(List<Integer> parentClassCount, List<List<Integer>> childClassCounts) {
        return gain(parentClassCount, childClassCounts, ScoreUtils::giniImpurityEntropyFromClassCounts);
    }

    public static double giniImpurity(Instances parentData, List<Instances> childData) {
        return giniImpurity(findClassCounts(parentData), findClassCounts(childData));
    }

    public static double giniImpurityEntropy(List<Instances> childData) {
        return 0.5 - entropy(findClassCounts(childData), ScoreUtils::giniImpurityEntropyFromClassCounts);
    }

    public static List<Integer> findClassCounts(Instances data) {
        TreeMap<Double, Integer> map = new TreeMap<>();
        for(Instance instance : data) {
            map.merge(instance.classValue(), 1, (a, b) -> a + b);
        }
        return new ArrayList<>(map.values());
    }

    public static List<List<Integer>> findClassCounts(List<Instances> datas) {
        return convert(datas, ScoreUtils::findClassCounts);
    }
}

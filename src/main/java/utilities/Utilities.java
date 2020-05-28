/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package utilities;

import com.beust.jcommander.internal.Lists;
import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.stopwatch.StopWatch;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.classifiers.distance_based.utils.collections.IntListView;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializedObject;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.regex.Pattern;

public class Utilities {

//    public static int add(int a, int b) {
//        int result = a + b;
//        if(result < a || result < b) {
//            throw new IllegalStateException("integer overflow");
//        }
//        return result;
//    }

    public static <A> int sum(Iterator<A> iterator, Function<A, Integer> func) {
        int sum = 0;
        while(iterator.hasNext()) {
            A next = iterator.next();
            Integer integer = func.apply(next);
            sum += integer;
        }
        return sum;
    }

    public static <A, B> List<B> convert(Iterable<A> source, Function<A, B> converter) { // todo stream version
        return convert(source.iterator(), converter);
    }

    public static <A, B> List<B> convert(Iterator<A> source, Function<A, B> converter) {
        return convert(source, converter, ArrayList::new);
    }

    public static <A, B, C extends Collection<B>> C convert(Iterator<A> source, Function<A, B> converter, Supplier<C> supplier) {
        C destination = supplier.get();
        while(source.hasNext()) {
            A item = source.next();
            B convertedItem = converter.apply(item);
            destination.add(convertedItem);
        }
        return destination;
    }

    public static <A, B, C extends Collection<B>> C convert(Iterable<A> source, Function<A, B> converter, Supplier<C> supplier) {
        return convert(source.iterator(), converter, supplier);
    }
//
//    public static void listenToTrainTimer(Object obj, StopWatch stated) {
//        if(obj instanceof TimedTrainAndTrainEstimate) {
//            try {
//                StopWatch trainTimer = ((TimedTrainAndTrainEstimate) obj).getTrainTimer();
//                trainTimer.addListener(stated);
//            } catch (UnsupportedOperationException ignored) {}
//        }
//    }
//
//    public static void listenToTrainEstimateTimer(Object obj, StopWatch stated) {
//        if(obj instanceof TimedTrainAndTrainEstimate) {
//            try {
//                StopWatch trainTimer = ((TimedTrainAndTrainEstimate) obj).getTrainEstimateTimer();
//                trainTimer.addListener(stated);
//            } catch (UnsupportedOperationException ignored) {}
//        }
//    }
//
//    public static void listenToMemoryWatcher(Object obj, MemoryWatcher stated) {
//        if(obj instanceof WatchedMemory) {
//            try {
//                MemoryWatcher memoryWatcher = ((WatchedMemory) obj).getMemoryWatcher();
//                memoryWatcher.addListener(stated);
//            } catch (UnsupportedOperationException ignored) {}
//        }
//    }
//
//    public static void unListenFromTrainTimer(Object obj, StopWatch stated) {
//        if(obj instanceof TimedTrainAndTrainEstimate) {
//            try {
//                StopWatch trainTimer = ((TimedTrainAndTrainEstimate) obj).getTrainTimer();
//                trainTimer.removeListener(stated);
//            } catch (UnsupportedOperationException ignored) {}
//        }
//    }
//
//    public static void unListenFromTrainEstimateTimer(Object obj, StopWatch stated) {
//        if(obj instanceof TimedTrainAndTrainEstimate) {
//            try {
//                StopWatch trainTimer = ((TimedTrainAndTrainEstimate) obj).getTrainEstimateTimer();
//                trainTimer.removeListener(stated);
//            } catch (UnsupportedOperationException ignored) {}
//        }
//    }
//
//    public static void unListenFromMemoryWatcher(Object obj, MemoryWatcher stated) {
//        if(obj instanceof WatchedMemory) {
//            try {
//                MemoryWatcher memoryWatcher = ((WatchedMemory) obj).getMemoryWatcher();
//                memoryWatcher.removeListener(stated);
//            } catch (UnsupportedOperationException ignored) {}
//        }
//    }

    public static long toNanos(String amountStr, String unitStr) {
        long amount = Long.parseLong(amountStr);
        TimeUnit unit = TimeUnit.valueOf(unitStr);
        long timeInNanos = TimeUnit.NANOSECONDS.convert(amount, unit);
        return timeInNanos;
    }

    public static final int size(double[][] matrix) {
        int population = 0;
        for(int i = 0; i < matrix.length; i++) {
            population += matrix[i].length;
        }
        return population;
    }
/**

* 6/2/19: bug fixed so it properly ignores the class value, only place its used
* is in measures.DistanceMeasure
 * @param instance
 * @return array of doubles with the class value removed
*/
    public static final double[] extractTimeSeries(Instance instance) {
        if(instance.classIndex() < 0) {
            return instance.toDoubleArray();
        } else {
            double[] timeSeries = new double[instance.numAttributes() - 1];
            for(int i = 0; i < instance.numAttributes(); i++) {
                if(i < instance.classIndex()) {
                    timeSeries[i] = instance.value(i);
                } else if (i != instance.classIndex()){
                    timeSeries[i - 1] = instance.value(i);
                }
            }
            return timeSeries;
        }
    }

    public static final double min(double... values) {
        double min = values[0];
        for(int i = 1; i < values.length; i++) {
            min = Math.min(min, values[i]);
        }
        return min;
    }

    public static final double sum(double[] array, int start, int end) {
        double sum = 0;
        for(int i = start; i < end; i++) {
            sum += array[i];
        }
        return sum;
    }

    public static final double sum(double[] array) {
        return sum(array, 0, array.length);
    }

    public static final double[] normalise(double[] array) {
        return normalise(array, sum(array));
    }

    public static double[] normalise(double[] array, double against) {
        for(int i = 0; i < array.length; i++) {
            array[i] /= against;
        }
        return array;
    }

    public static final double[] normalisePercentage(double[] array) {
        return normalise(array, sum(array) / 100);
    }

    public static final String sanitiseFolderPath(String path) {
        if(path.charAt(path.length() - 1) != '/') {
            path = path + '/';
        }
        return path;
    }

    public static final double max(double... values) {
        double max = values[0];
        for(int i = 1; i < values.length; i++) {
            max = Math.max(max, values[i]);
        }
        return max;
    }

    public static final double[] divide(double[] a, double[] b) {
        double[] result = new double[a.length];
        for(int i = 0; i < result.length; i++) {
            result[i] = a[i] / b[i];
        }
        return result;
    }

    public static final double[] divide(double[] array, int divisor) {
        double[] result = new double[array.length];
        for(int i = 0; i < result.length; i++) {
            result[i] = array[i] / divisor;
        }
        return result;
    }

    public static final int maxIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if(array[maxIndex] < array[i]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static final int minIndex(double[] array) {
        int minIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if(array[i] < array[minIndex]) {
                minIndex = i;
            }
        }
        return minIndex;
    }

    public static final void zeroOrMore(int i) {
        if(i < 0) {
            throw new IllegalArgumentException("less than zero");
        }
    }

    public static final void moreThanOrEqualTo(int a, int b) {
        if(b < a) {
            throw new IllegalArgumentException("b cannot be less than a");
        }
    }

    public static double log(double value, double base) { // beware, this is inaccurate due to floating point error!
        if(value == 0) {
            return 0;
        }
        return Math.log(value) / Math.log(base);
    }

    public static Map<Double, Instances> instancesByClass(Instances instances) {
        Map<Double, Instances> map = new TreeMap<>();
        for(Instance instance : instances) {
            map.computeIfAbsent(instance.classValue(),  k -> new Instances(instances, 0)).add(instance);
        }
        return map;
    }

    public static double sum(Iterable<Double> iterable) {
        return sum(iterable, (Function<Double, Double>) aDouble -> aDouble);
    }

    public static <A> double sum(Iterable<A> iterable, Function<A, Double> func) {
        double sum = 0;
        for(A item : iterable) {
            sum += func.apply(item);
        }
        return sum;
    }

    public static List<Double> divide(Iterable<Integer> iterable, int quotient) {
        List<Double> result = Lists.newArrayList();
        for(Integer integer : iterable) {
            double v = (double) integer / quotient;
            result.add(v);
        }
        return result;
    }

    public static List<Double> normalise(Iterable<Integer> iterable) {
        List<Integer> list = ArrayUtilities.drain(iterable);
        return divide(list, ArrayUtilities.sum(list));
    }

    public static <A, B> boolean isUnique(final Iterator<A> iterator, Function<A, B> func) {
        if(!iterator.hasNext()) {
            return true;
        }
        B value = func.apply(iterator.next());
        while(iterator.hasNext()) {
            B nextValue = func.apply(iterator.next());
            if(value == null) {
                if(nextValue != null) {
                    return false;
                }
            } else if(!value.equals(nextValue)) {
                return false;
            }
        }
        return true;
    }

    public static <A> boolean isUnique(final Iterator<A> iterator) {
        return isUnique(iterator, i -> i);
    }

    public static <A> boolean isUnique(Iterable<A> iterable) {
        return isUnique(iterable.iterator());
    }

    public static <A, B> boolean isUnique(Iterable<A> iterable, Function<A, B> func) {
        return isUnique(iterable.iterator(), func);
    }

    public static boolean isHomogeneous(Instances data) {
        return isUnique(data, Instance::classValue);
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
                distribution = divide(getCounts(), sum);
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
            double score = proportion * log(proportion, 2);
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
        for(List<Integer> classCount : childClassCounts) {
            // find the proportion of classes in the child node
            ClassCount childClassCount = new ClassCount(classCount);
            double proportion = (double) childClassCount.getSum() / parentClassCountSum;
            // find the entropy at the child
            double entropy = entropyFunction.apply(childClassCount);
            // weight the entropy by the number of cases at the child node
            entropy *= proportion;
            // subtract the child entropy from the parent for relative improvement
            score -= entropy;
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
        return gain(parentClassCount, childClassCounts, Utilities::infoGainEntropyFromClassCounts);
    }

    public static double infoGain(Instances parentData, List<Instances> childData) {
        return infoGain(findClassCounts(parentData), findClassCounts(childData));
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
        return gain(parentClassCount, childClassCounts, Utilities::giniImpurityEntropyFromClassCounts);
    }

    public static double giniImpurity(Instances parentData, List<Instances> childData) {
        return giniImpurity(findClassCounts(parentData), findClassCounts(childData));
    }

    public static List<Integer> findClassCounts(Instances data) {
        TreeMap<Double, Integer> map = new TreeMap<>();
        for(Instance instance : data) {
            map.merge(instance.classValue(), 1, (a, b) -> a + b);
        }
        return new ArrayList<>(map.values());
    }

    public static List<List<Integer>> findClassCounts(List<Instances> datas) {
        return convert(datas, Utilities::findClassCounts);
    }

    public static class UnitTests {

        @Test
        public void testGiniImpurityEntropyImpure() {
            double score = Utilities.giniImpurityEntropyFromClassCounts(Lists.newArrayList(5, 5));
            System.out.println(score);
            Assert.assertTrue(score == 0.5);
        }

        @Test
        public void testGiniImpurityEntropyPure() {
            double score = Utilities.giniImpurityEntropyFromClassCounts(Lists.newArrayList(10, 0));
            System.out.println(score);
            Assert.assertTrue(score == 0);
        }

        @Test
        public void testGiniImpurityEntropyA() {
            double score = Utilities.giniImpurityEntropyFromClassCounts(Lists.newArrayList(4, 6));
            System.out.println(score);
            Assert.assertTrue(score == 0.48);
        }

        @Test
        public void testGiniImpurityEntropyB() {
            double score = Utilities.giniImpurityEntropyFromClassCounts(Lists.newArrayList(1, 9));
            System.out.println(score);
            Assert.assertTrue(score == 0.17999999999999994);
        }

        @Test
        public void testGiniImpurityPure() {
            double score = giniImpurity(Lists.newArrayList(8, 4), Lists.newArrayList(
                Lists.newArrayList(8, 0),
                Lists.newArrayList(0, 4)
            ));
            System.out.println(score);
            Assert.assertTrue(score == 0.4444444444444444);
        }

        @Test
        public void testGiniImpurityImpure() {
            double score = giniImpurity(Lists.newArrayList(8, 4), Lists.newArrayList(
                Lists.newArrayList(4, 2),
                Lists.newArrayList(4, 2)
            ));
            System.out.println(score);
            Assert.assertTrue(score == 0);
        }

        @Test
        public void testGiniImpurityA() {
            double score = giniImpurity(Lists.newArrayList(8, 4), Lists.newArrayList(
                Lists.newArrayList(7, 3),
                Lists.newArrayList(1, 1)
            ));
            System.out.println(score);
            Assert.assertTrue(score == 0.011111111111111058);
        }

        @Test
        public void testGiniImpurityB() {
            double score = giniImpurity(Lists.newArrayList(8, 4), Lists.newArrayList(
                Lists.newArrayList(7, 1),
                Lists.newArrayList(1, 3)
            ));
            System.out.println(score);
            Assert.assertTrue(score == 0.1736111111111111);
        }

        @Test
        public void testInfoGainEntropyImpure() {
            double score = infoGainEntropyFromClassCounts(Lists.newArrayList(5, 5));
            System.out.println(score);
            Assert.assertTrue(score == 1);
        }

        @Test
        public void testInfoGainEntropyPure() {
            double score = infoGainEntropyFromClassCounts(Lists.newArrayList(10, 0));
            System.out.println(score);
            Assert.assertTrue(score == 0);
        }

        @Test
        public void testInfoGainPure() {
            double score = infoGain(Lists.newArrayList(8, 4), Lists.newArrayList(
                Lists.newArrayList(8, 0),
                Lists.newArrayList(0, 4)
            ));
            System.out.println(score);
            Assert.assertTrue(score == 0.9182958340544896);
        }

        @Test
        public void testInfoGainImpure() {
            double score = infoGain(Lists.newArrayList(8, 4), Lists.newArrayList(
                Lists.newArrayList(4, 2),
                Lists.newArrayList(4, 2)
            ));
            System.out.println(score);
            Assert.assertTrue(score == 0);
        }

        @Test
        public void testInfoGainA() {
            double score = infoGain(Lists.newArrayList(8, 4), Lists.newArrayList(
                Lists.newArrayList(7, 3),
                Lists.newArrayList(1, 1)
            ));
            System.out.println(score);
            Assert.assertTrue(score == 0.01722008469557898);
        }

        @Test
        public void testInfoGainB() {
            double score = infoGain(Lists.newArrayList(8, 4), Lists.newArrayList(
                Lists.newArrayList(7, 1),
                Lists.newArrayList(1, 3)
            ));
            System.out.println(score);
            Assert.assertTrue(score == 0.28549349710171434);
        }
    }

    public static final double[] interpolate(double min, double max, int num) {
        double[] result = new double[num];
        double diff = (max - min) / (num - 1);
        for(int i = 0; i < result.length; i++) {
            result[i] = min + diff * i;
        }
        return result;
    }

    public static int[] argMax(double[] array) {
        List<Integer> indices = new ArrayList<>();
        double max = array[0];
        indices.add(0);
        for(int i = 1; i < array.length; i++) {
            if(array[i] >= max) {
                if(array[i] > max) {
                    max = array[i];
                    indices.clear();
                }
                indices.add(i);
            }
        }
        int[] indicesCopy = new int[indices.size()];
        for(int i = 0; i < indicesCopy.length; i++) {
            indicesCopy[i] = indices.get(i);
        }
        return indicesCopy;
    }

    public static int argMax(double[] array, Random random) {
        int[] indices = argMax(array);
        if(indices.length == 1) {
            return indices[0];
        }
        return indices[random.nextInt(indices.length)];
    }

    public static int[] fromPermutation(int permutataion, int... binSizes) {
        int maxCombination = numPermutations(binSizes) - 1;
        if(permutataion > maxCombination || binSizes.length == 0 || permutataion < 0) {
            throw new IllegalArgumentException();
        }
        int[] result = new int[binSizes.length];
        for(int index = 0; index < binSizes.length; index++) {
            int binSize = binSizes[index];
            if(binSize > 1) {
                result[index] = permutataion % binSize;
                permutataion /= binSize;
            } else {
                result[index] = 0;
                if(binSize <= 0) {
                    throw new IllegalArgumentException();
                }
            }
        }
        return result;
    }

    public static List<Integer> fromPermutation(int permutation, List<Integer> binSizes) {
        int maxCombination = numPermutations(binSizes) - 1;
        if(permutation > maxCombination || binSizes.size() == 0 || permutation < 0) {
            throw new IndexOutOfBoundsException();
        }
        List<Integer> result = new ArrayList<>();
        for(int binSize : binSizes) {
            if(binSize > 1) {
                result.add(permutation % binSize);
                permutation /= binSize;
            } else if(binSize == 1) {
                result.add(0);
            } else {
                // binSize is 0 or less (i.e. no index as that bin cannot be indexed as size <=0)
                result.add(-1);
            }
        }
        return result;
    }

    public static int toPermutation(int[] values, int[] binSizes) {
        return toPermutation(new IntListView(values), new IntListView(binSizes));
    }

    public static int toPermutation(List<Integer> values, List<Integer> binSizes) {
        if(values.size() != binSizes.size()) {
            throw new IllegalArgumentException("incorrect number of args");
        }
        int permutation = 0;
        for(int i = binSizes.size() - 1; i >= 0; i--) {
            int binSize = binSizes.get(i);
            if(binSize > 1) {
                int value = values.get(i);
                permutation *= binSize;
                permutation += value;
            } else if(binSize < 0){
                throw new IllegalArgumentException();
            }
        }
        return permutation;
    }

    public static int numPermutations(List<Integer> binSizes) {
        if(binSizes.isEmpty()) {
            return 0;
        }
        List<Integer> maxValues = new ArrayList<>();
        for(Integer binSize : binSizes) {
            int size = binSize - 1;
            if(size < 0) {
                size = 0;
            }
            maxValues.add(size);
        }
        return toPermutation(maxValues, binSizes) + 1;
    }

    public static int numPermutations(int[] binSizes) {
        return numPermutations(new IntListView(binSizes));
    }


    public static <A> List<A> asList(A[] array) {
        return Arrays.asList(array);
    }

    public static List<Integer> asList(int[] array) {
        return new IntListView(array);
    }

    public static List<Double> asList(double[] array) {
        return Arrays.asList(box(array));
    }
    public static List<Float> asList(float[] array) {
        return Arrays.asList(box(array));
    }
    public static List<Long> asList(long[] array) {
        return Arrays.asList(box(array));
    }
    public static List<Short> asList(short[] array) {
        return Arrays.asList(box(array));
    }
    public static List<Byte> asList(byte[] array) {
        return Arrays.asList(box(array));
    }
    public static List<Boolean> asList(boolean[] array) {
        return Arrays.asList(box(array));
    }
    public static List<Character> asList(char[] array) {
        return Arrays.asList(box(array));
    }


    // todo get rid of box uses and replace with listviews
    public static Integer[] box(int[] array) {
        Integer[] boxed = new Integer[array.length];
        for(int i = 0; i < array.length; i++) {
            boxed[i] = array[i];
        }
        return boxed;
    }
    public static Long[] box(long[] array) {
        Long[] boxed = new Long[array.length];
        for(int i = 0; i < array.length; i++) {
            boxed[i] = array[i];
        }
        return boxed;
    }
    public static Double[] box(double[] array) {
        Double[] boxed = new Double[array.length];
        for(int i = 0; i < array.length; i++) {
            boxed[i] = array[i];
        }
        return boxed;
    }
    public static Float[] box(float[] array) {
        Float[] boxed = new Float[array.length];
        for(int i = 0; i < array.length; i++) {
            boxed[i] = array[i];
        }
        return boxed;
    }
    public static Short[] box(short[] array) {
        Short[] boxed = new Short[array.length];
        for(int i = 0; i < array.length; i++) {
            boxed[i] = array[i];
        }
        return boxed;
    }
    public static Boolean[] box(boolean[] array) {
        Boolean[] boxed = new Boolean[array.length];
        for(int i = 0; i < array.length; i++) {
            boxed[i] = array[i];
        }
        return boxed;
    }
    public static Byte[] box(byte[] array) {
        Byte[] boxed = new Byte[array.length];
        for(int i = 0; i < array.length; i++) {
            boxed[i] = array[i];
        }
        return boxed;
    }
    public static Character[] box(char[] array) {
        Character[] boxed = new Character[array.length];
        for(int i = 0; i < array.length; i++) {
            boxed[i] = array[i];
        }
        return boxed;
    }

    public static Instances toInstances(Instance... instances) {
        Instances result = new Instances(instances[0].dataset(), 0);
        result.addAll(Utilities.asList(instances));
        return result;
    }

    public static Map<Double, Integer> classDistribution(Instances data) {
        Map<Double, Integer> distribution = new HashMap<>();
        for(Instance instance : data) {
            double classValue = instance.classValue();
            distribution.compute(classValue, (k, v) -> {
                if(v == null) {
                    return 1;
                } else {
                    return v + 1;
                }
            });
        }
        return distribution;
    }

    public static <A> Map<A, Double> normalise(Map<A, Integer> map) {
        Map<A, Double> distribution = new HashMap<>();
        int sum = ArrayUtilities.sum(map.values());
        for(Map.Entry<A, Integer> entry : map.entrySet()) {
            distribution.put(entry.getKey(), ((double) entry.getValue()) / sum);
        }
        return distribution;
    }

    @SuppressWarnings("unchecked")
    public static <A> A deepCopy(A value) throws Exception {
        if(value instanceof Serializable) {
            return (A) new SerializedObject(value).getObject();
        } else {
            String str = StrUtils.toOptionValue(value);
            Object object = StrUtils.fromOptionValue(str);
            return (A) object;
        }
    }

    public static <A> A randPickOne(Collection<A> collection, Random random) {
        List<A> list = randPickN(collection, 1, random);
        if(list.size() != 1) {
            throw new IllegalStateException("was expecting only 1 result");
        }
        return list.get(0);
    }

    public static <A> List<A> randPickN(Collection<A> collection, int num, Random random) {
        Assert.assertNotNull(collection);
        Assert.assertNotNull(random);
        if(num > collection.size()) {
            throw new IllegalArgumentException("too many");
        }
        if(num < 0) {
            throw new IllegalArgumentException("num is neg");
        }
        List<A> list = new ArrayList<>(collection);
        if(num == collection.size()) {
            return list;
        }
        List<A> removed = new ArrayList<>();
        for(int i = 0; i < num; i++) {
            int index = random.nextInt(list.size());
            A removedItem = list.remove(index);
            removed.add(removedItem);
        }
        return removed;
    }
   
    public static boolean stringIsDouble(String input){
         /*********Aarons Nasty Regex From https://docs.oracle.com/javase/8/docs/api/java/lang/Double.html#valueOf-java.lang.String **********/
        final String Digits     = "(\\p{Digit}+)";
        final String HexDigits  = "(\\p{XDigit}+)";
        // an exponent is 'e' or 'E' followed by an optionally
        // signed decimal integer.
        final String Exp        = "[eE][+-]?"+Digits;
        final String fpRegex    =
            ("[\\x00-\\x20]*"+  // Optional leading "whitespace"
            "[+-]?(" + // Optional sign character
            "NaN|" +           // "NaN" string
            "Infinity|" +      // "Infinity" string
    
            // A decimal floating-point string representing a finite positive
            // number without a leading sign has at most five basic pieces:
            // Digits . Digits ExponentPart FloatTypeSuffix
            //
            // Since this method allows integer-only strings as input
            // in addition to strings of floating-point literals, the
            // two sub-patterns below are simplifications of the grammar
            // productions from section 3.10.2 of
            // The Java Language Specification.
    
            // Digits ._opt Digits_opt ExponentPart_opt FloatTypeSuffix_opt
            "((("+Digits+"(\\.)?("+Digits+"?)("+Exp+")?)|"+
    
            // . Digits ExponentPart_opt FloatTypeSuffix_opt
            "(\\.("+Digits+")("+Exp+")?)|"+
    
            // Hexadecimal strings
            "((" +
            // 0[xX] HexDigits ._opt BinaryExponent FloatTypeSuffix_opt
            "(0[xX]" + HexDigits + "(\\.)?)|" +
    
            // 0[xX] HexDigits_opt . HexDigits BinaryExponent FloatTypeSuffix_opt
            "(0[xX]" + HexDigits + "?(\\.)" + HexDigits + ")" +
    
            ")[pP][+-]?" + Digits + "))" +
            "[fFdD]?))" +
            "[\\x00-\\x20]*");// Optional trailing "whitespace"

        return Pattern.matches(fpRegex, input);
    }



}

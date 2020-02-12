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

import tsml.classifiers.GcMemoryWatchable;
import tsml.classifiers.MemoryWatchable;
import tsml.classifiers.StopWatchTrainTimeable;
import tsml.classifiers.TrainTimeable;
import tsml.classifiers.distance_based.knn.KnnLoocv;
import utilities.collections.IntListView;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializedObject;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

public class Utilities {


    public static void listenToTrainTimer(Object obj, StopWatch stated) {
        if(obj instanceof StopWatchTrainTimeable) {
            try {
                StopWatch trainTimer = ((StopWatchTrainTimeable) obj).getTrainTimer();
                trainTimer.addListener(stated);
            } catch (UnsupportedOperationException ignored) {}
        }
    }

    public static void listenToTrainEstimateTimer(Object obj, StopWatch stated) {
        if(obj instanceof StopWatchTrainTimeable) {
            try {
                StopWatch trainTimer = ((StopWatchTrainTimeable) obj).getTrainEstimateTimer();
                trainTimer.addListener(stated);
            } catch (UnsupportedOperationException ignored) {}
        }
    }

    public static void listenToMemoryWatcher(Object obj, MemoryWatcher stated) {
        if(obj instanceof GcMemoryWatchable) {
            try {
                MemoryWatcher memoryWatcher = ((GcMemoryWatchable) obj).getMemoryWatcher();
                memoryWatcher.addListener(stated);
            } catch (UnsupportedOperationException ignored) {}
        }
    }

    public static void unListenFromTrainTimer(Object obj, StopWatch stated) {
        if(obj instanceof StopWatchTrainTimeable) {
            try {
                StopWatch trainTimer = ((StopWatchTrainTimeable) obj).getTrainTimer();
                trainTimer.removeListener(stated);
            } catch (UnsupportedOperationException ignored) {}
        }
    }

    public static void unListenFromTrainEstimateTimer(Object obj, StopWatch stated) {
        if(obj instanceof StopWatchTrainTimeable) {
            try {
                StopWatch trainTimer = ((StopWatchTrainTimeable) obj).getTrainEstimateTimer();
                trainTimer.removeListener(stated);
            } catch (UnsupportedOperationException ignored) {}
        }
    }

    public static void unListenFromMemoryWatcher(Object obj, MemoryWatcher stated) {
        if(obj instanceof GcMemoryWatchable) {
            try {
                MemoryWatcher memoryWatcher = ((GcMemoryWatchable) obj).getMemoryWatcher();
                memoryWatcher.removeListener(stated);
            } catch (UnsupportedOperationException ignored) {}
        }
    }

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
        if(instance.classIsMissing()) {
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

    public static final double log(double value, double base) { // beware, this is inaccurate due to floating point error!
        return Math.log(value) / Math.log(base);
    }


    public static double giniScore(int parent, int... children) {
        if(parent <= 0) {
            throw new IllegalArgumentException("parent leq 0");
        }
        int sum = ArrayUtilities.sum(children);
        if(sum > parent) {
            throw new IllegalArgumentException("children sum greater than parent");
        }
        double scoreSum = 0;
        for(int child : children) {
            double proportion = (double) child / parent;
            double score = Math.pow(child, 2);
            score *= proportion;
            scoreSum += score;
        }
        return 1 - scoreSum;
    }

    public static Map<Double, Instances> instancesByClass(Instances instances) {
        Map<Double, Instances> map = new HashMap<>();
        for(Instance instance : instances) {
            map.computeIfAbsent(instance.classValue(),  k -> new Instances(instances, 0)).add(instance);
        }
        return map;
    }

    public static double informationGain(int parent, int... children) {
        if(parent <= 0) {
            throw new IllegalArgumentException("parent leq 0");
        }
        int sum = ArrayUtilities.sum(children);
        if(sum > parent) {
            throw new IllegalArgumentException("children sum greater than parent");
        }
        double scoreSum = 0;
        for(int child : children) {
            double score = child * Utilities.log(child, 2);
            scoreSum += score;
        }
        return 0 - scoreSum;
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
            } else {
                if(binSize < 0) {
                    throw new IllegalArgumentException();
                }
                result.add(binSize - 1);
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

    public static int sum(Iterator<Integer> iterator) {
        int sum = 0;
        while(iterator.hasNext()) {
            sum += iterator.next();
        }
        return sum;
    }

    public static int sum(Iterable<Integer> iterable) {
        return sum(iterable.iterator());
    }

    public static <A> Map<A, Double> normalise(Map<A, Integer> map) {
        Map<A, Double> distribution = new HashMap<>();
        int sum = sum(map.values());
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

    public static <A> List<A> randPickN(Collection<A> collection, int num, Random rand) {
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
            int index = rand.nextInt(list.size());
            A removedItem = list.remove(index);
            removed.add(removedItem);
        }
        return removed;
    }

    public static <A, B> List<B> convert(Iterable<A> iterable, Function<A, B> func) {
        return convert(iterable.iterator(), func);
    }

    public static <A, B> List<B> convert(Iterator<A> iterator, Function<A, B> func) {
        List<B> list = new ArrayList<>();
        while(iterator.hasNext()) {
            A item = iterator.next();
            B convertedItem = func.apply(item);
            list.add(convertedItem);
        }
        return list;
    }
}

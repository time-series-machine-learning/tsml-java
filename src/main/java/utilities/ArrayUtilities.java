package utilities;

import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.function.Function;

public class ArrayUtilities {
    private ArrayUtilities() {}

    public static void addInPlace(double[] a, double[] b) {
        if(a.length < b.length) {
            throw new IllegalArgumentException();
        }
        for(int i = 0; i < b.length; i++) {
            a[i] += b[i];
        }
    }

    public static void subtract(double[] a, double[] b) {
        int length = Math.min(a.length, b.length);
        for(int i = 0; i < length; i++) {
            a[i] -= b[i];
        }
    }

    public static double sum(double[] array) {
        double sum = 0;
        for(int i = 0; i < array.length; i++) {
            sum += array[i];
        }
        return sum;
    }

    public static void normaliseInPlace(double[] array) {
        double sum = sum(array);
        if(sum == 0) {
            throw new IllegalArgumentException("sum of zero");
        }
        for(int i = 0; i < array.length; i++) {
            array[i] /= sum;
        }
    }

    public static void multiplyInPlace(double[] array, double multiplier) {
        for(int i = 0; i < array.length; i++) {
            array[i] *= multiplier;
        }
    }

    public static int[] maxIndex(double[] array) {
        List<Integer> indices = new ArrayList<>();
        indices.add(0);
        double max = array[0];
        for(int i = 1; i < array.length; i++) {
            double value = array[i];
            if(value >= max) {
                if(value > max) {
                    max = value;
                    indices.clear();
                }
                indices.add(i);
            }
        }
        int[] result = new int[indices.size()];
        for(int i = 0; i < result.length; i++) {
            result[i] = indices.get(i);
        }
        return result;
    }

    public static double mean(double[] array) {
        return sum(array) / array.length;
    }

    public static double populationStandardDeviation(Instances input) {
        double sumx = 0;
        double sumx2 = 0;
        double[] ins2array;
        for(int i = 0; i < input.numInstances(); i++){
            ins2array = input.instance(i).toDoubleArray();
            for(int j = 0; j < ins2array.length-1; j++){//-1 to avoid classVal
                sumx+=ins2array[j];
                sumx2+=ins2array[j]*ins2array[j];
            }
        }
        int n = input.numInstances()*(input.numAttributes()-1);
        double mean = sumx/n;
        return Math.sqrt(sumx2/(n)-mean*mean);
    }

    public static double[] derivative(double[] array) {
        double[] derivative = new double[array.length - 1];
        for(int i = 0; i < derivative.length; i++) {
            derivative[i] = array[i + 1] - array[i];
        }
        return derivative;
    }

    public static int gcd(int a, int b) {
        while (b != 0) {
            int temp = a;
            a = b;
            b = temp % b;
        }
        return a;
    }

    public static int gcd(int... values) {
        int result = values[0];
        for(int value : values) {
            result = gcd(result, value);
        }
        return result;
    }

    public static void divideGcd(int[] array) {
        int gcd = gcd(array);
        divide(array, gcd);
    }

    public static void divide(int[] array, int divisor) {
        for(int i = 0; i < array.length; i++) {
            array[i] /= divisor;
        }
    }

    public static void main(String[] args) {
        int[] abc = new int[] {120,268,782};
        divideGcd(abc);
        for(int i : abc) {
            System.out.print(i);
            System.out.print(", ");
        }
    }

    // don't use unless you want imprecision due to incrementation
    public static int[] incrementalRange(int min, int max, int size){
        int[] output = new int[size];

        double diff = (double)(max-min)/(size - 1);
        double[] doubleOut = new double[size];
        doubleOut[0] = min;
        output[0] = min;
        for(int i = 1; i < size - 1; i++){
            doubleOut[i] = doubleOut[i-1]+diff;
            output[i] = (int)Math.round(doubleOut[i]);
        }
        output[size - 1] = max;
        return output;
    }

    // don't use unless you want imprecision due to incrementation
    public static double[] incrementalRange(double min, double max, int size){
        double[] output = new double[size];
        double diff = (max-min)/(size - 1);
        output[0] = min;
        for(int i = 1; i < size - 1; i++){
            output[i] = output[i-1]+diff;
        }
        output[size - 1] = max; // to make sure max isn't omitted due to double imprecision
        return output;
    }

    public static <A> void removeDuplicatesInPlace(List<A> values) {
        Set<A> set = new TreeSet<>(values); // must be treeset to maintain ordering
        values.clear();
        values.addAll(set);
    }


    public static <T> T[] concat(T[] first, T[]... rest) {
        int totalLength = first.length;
        for (T[] array : rest) {
            totalLength += array.length;
        }
        T[] result = Arrays.copyOf(first, totalLength);
        int offset = first.length;
        for (T[] array : rest) {
            System.arraycopy(array, 0, result, offset, array.length);
            offset += array.length;
        }
        return result;
    }

    public static Instances toInstances(Instance... instances) {
        Instances collection = new Instances(instances[0].dataset(), 0);
        collection.addAll(Arrays.asList(instances));
        return collection;
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
            throw new IllegalArgumentException();
        }
        List<Integer> result = new ArrayList<>();
        for(int index = 0; index < binSizes.size(); index++) {
            int binSize = binSizes.get(index);
            if(binSize > 1) {
                result.add(permutation % binSize);
                permutation /= binSize;
            } else {
                result.add(0);
                if(binSize <= 0) {
                    throw new IllegalArgumentException();
                }
            }
        }
        return result;
    }

    public static int toPermutation(int[] values, int[] binSizes) {
        return toPermutation(primitiveArrayToList(values), primitiveArrayToList(binSizes));
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
            } else if(binSize <= 0) {
                throw new IllegalArgumentException();
            }
        }
        return permutation;
    }


    public static int numPermutations(List<Integer> binSizes) {
        List<Integer> maxValues = new ArrayList<>();
        for(int i = 0; i < binSizes.size(); i++) {
            maxValues.add(binSizes.get(i) - 1);
        }
        return toPermutation(maxValues, binSizes) + 1;
    }

    public static int numPermutations(int[] binSizes) {
        return numPermutations(primitiveArrayToList(binSizes));
    }

    public static List<Integer> primitiveArrayToList(int[] values) {
        List<Integer> list = new ArrayList<>();
        for(int i = 0; i < values.length; i++) {
            list.add(i);
        }
        return list;
    }


    public static <A extends List<Integer>> A sequence(int j, A list) {
        for(int i = 0; i < j; i++) {
            list.add(i);
        }
        return list;
    }

    public static List<Integer> sequence(int j) {
        return sequence(j, new ArrayList<>());
    }

    public static <A> List<A> flatten(Map<?, List<A>> map) {
        List<A> list = new ArrayList<>();
        for (Map.Entry<?, List<A>> entry : map.entrySet()) {
            list.addAll(entry.getValue());
        }
        return list;
    }

    public static <A> List<Integer> bestIndices(List<A> list, Comparator<A> comparator) {
        List<Integer> indices = new ArrayList<>();
        if(!list.isEmpty()) {
            indices.add(0);
            for(int i = 0; i < list.size(); i++) {
                A item = list.get(i);
                int comparison = comparator.compare(item, list.get(indices.get(0)));
                if(comparison >= 0) {
                    if(comparison > 0) {
                        indices.clear();
                    }
                    indices.add(i);
                }
            }
        }
        return indices;
    }

    public static <A> int bestIndex(List<A> list, Comparator<A> comparator, Random random) {
        List<Integer> indices = bestIndices(list, comparator);
        return indices.get(random.nextInt(indices.size()));
    }

    public static <A extends Comparable<A>> int bestIndex(List<A> list, Random random) {
        return bestIndex(list, Comparable::compareTo, random);
    }


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


}

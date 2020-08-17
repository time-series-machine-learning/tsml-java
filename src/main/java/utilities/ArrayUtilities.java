package utilities;

import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;

public class ArrayUtilities {
    private ArrayUtilities() {}

    public static double[][] transposeMatrix(double [][] m){
        double[][] temp = new double[m[0].length][m.length];
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[0].length; j++)
                temp[j][i] = m[i][j];
        return temp;
    }

    public static double[] intToDouble(int[] input) {
        double[] output = new double[input.length];
        for(int i = 0; i < output.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    public static double[][] intToDouble(int[][] input) {
        double[][] output = new double[input.length][];
        for(int i = 0; i < output.length; i++) {
            output[i] = intToDouble(input[i]);
        }
        return output;
    }

    public static double[] oneHot(int length, int index) {
        final double[] array = new double[length];
        array[index] = 1;
        return array;
    }

    public static void add(double[] src, double[] addend) {
        if(src.length < addend.length) {
            throw new IllegalArgumentException();
        }
        for(int i = 0; i < addend.length; i++) {
            src[i] += addend[i];
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

    public static double[] normalise(double[] array, boolean ignoreZeroSum) {
        double sum = sum(array);
        if(sum == 0) {
            if(ignoreZeroSum) {
                return uniformDistribution(array.length);
            }
            throw new IllegalArgumentException("sum of zero");
        }
        for(int i = 0; i < array.length; i++) {
            array[i] /= sum;
        }
        return array;
    }

    public static double[] copy(double[] input) {
        double[] output = new double[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return output;
    }

    public static double[] normalise(double[] array) {
        return normalise(array, true);
    }

    public static double[] normalise(int[] array) {
        return normalise(array, true);
    }

    public static double[] normalise(int[] array, boolean ignoreZeroSum) {
        double sum = sum(array);
        double[] result = new double[array.length];
        if(sum == 0 && !ignoreZeroSum) {
            throw new IllegalArgumentException("sum of zero");
        }
        for(int i = 0; i < array.length; i++) {
            result[i] = (double) array[i] / sum;
        }
        return result;
    }

    public static <A> List<A> drain(Iterable<A> iterable) {
        return drain(iterable.iterator());
    }

    public static <A> List<A> drain(Iterator<A> iterator) {
        List<A> list = new ArrayList<>();
        while(iterator.hasNext()) {
            list.add(iterator.next());
        }
        return list;
    }

    public static double sum(List<Double> list) {
        return list.stream().reduce(0d, Double::sum);
    }

    public static List<Double> normalise(List<Double> list) {
        double sum = sum(list);
        if(sum == 0) {
            throw new IllegalArgumentException("sum zero");
        }
        return list.stream().map(element -> element / sum).collect(Collectors.toList());
    }

    public static List<Double> normalise(Iterable<Double> iterable) {
        List<Double> list = drain(iterable);
        return normalise(list);
    }

    public static void multiply(double[] array, double multiplier) {
        for(int i = 0; i < array.length; i++) {
            array[i] *= multiplier;
        }
    }

    public static double mean(double[] array) {
        return sum(array) / array.length;
    }

    public static void divide(int[] array, int divisor) {
        for(int i = 0; i < array.length; i++) {
            array[i] /= divisor;
        }
    }

    public static List<Object> asList(Object[] array) {
        return Arrays.asList(array);
    }
    public static List<Integer> asList(int[] array) {
        return Arrays.asList(box(array));
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

    public static double[] unbox(List<Double> list) {
        double[] array = new double[list.size()];
        for(int i = 0; i < array.length; i++) {
            array[i] = list.get(i);
        }
        return array;
    }

    public static int[] range(int min, int max, int size) {
        int[] output = new int[size];
        output[0] = min;
        output[size - 1] = max;
        for(int i = 1; i < size - 1; i++) {
            output[i] = (int) Math.round(((double) (max - min)) / (size - 1) * i);
        }
        return output;
    }

    public static double[] range(double min, double max, int size) {
        double[] output = new double[size];
        output[0] = min;
        output[size - 1] = max;
        for(int i = 1; i < size - 1; i++) {
            output[i] = (max - min) / (size - 1) * i + min;
        }
        return output;
    }

    public static <A extends Comparable<A>> List<A> unique(Collection<A> values) {
        return unique(new ArrayList<>(values), Comparator.naturalOrder());
    }

    public static <A extends Comparable<A>> List<A> unique(List<A> values) {
        return unique(values, Comparator.naturalOrder());
    }

    public static List<Double> unique(double[] values) {
        return unique(new ArrayList<>(asList(values)));
    }

    public static List<Integer> unique(int[] values) {
        return unique(new ArrayList<>(asList(values)));
    }

    public static <A> List<A> unique(List<A> values, Comparator<A> comparator) {
        Set<A> set = new TreeSet<>(comparator); // must be treeset to maintain ordering
        set.addAll(values);
        values.clear();
        values.addAll(set);
        return values;
    }

    public static List<Integer> fromPermutation(int permutation, List<Integer> binSizes) {
        int maxCombination = numPermutations(binSizes) - 1;
        if(permutation > maxCombination || binSizes.size() == 0 || permutation < 0) {
            throw new IndexOutOfBoundsException();
        }
        List<Integer> result = new ArrayList<>();
        for(int index = 0; index < binSizes.size(); index++) {
            int binSize = binSizes.get(index);
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
        for(int i = 0; i < binSizes.size(); i++) {
            int size = binSizes.get(i) - 1;
            if(size < 0) {
                size = 0;
            }
            maxValues.add(size);
        }
        return toPermutation(maxValues, binSizes) + 1;
    }

    /**
     * produce a list of ints from 0 to limit - 1 (inclusively)
     * @param j the limit
     * @return
     */
    public static <A extends List<Integer>> A sequence(int j, A list) {
        for(int i = 0; i < j; i++) {
            list.add(i);
        }
        return list;
    }

    public static List<Integer> sequence(int j) {
        return sequence(j, new ArrayList<>(j));
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

    public static String toString(int[][] matrix, String horizontalSeparator, String verticalSeparator) {
        StringBuilder builder = new StringBuilder();
        for(int i = 0; i < matrix.length; i++) {
            for(int j = 0; j < matrix[i].length; j++) {
//                builder.append(new BigDecimal(matrix[i][j]).setScale(2, RoundingMode.HALF_UP).doubleValue());
                builder.append(matrix[i][j]);
                if(j != matrix[i].length - 1) {
                    builder.append(horizontalSeparator);
                }
            }
            if(i != matrix.length - 1) {
                builder.append(verticalSeparator);
            }
        }
        builder.append(System.lineSeparator());
        return builder.toString();
    }

    public static int sum(final int... nums) {
        int sum = 0;
        for(int num : nums) {
            sum += num;
        }
        return sum;
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

    public static double[] uniformDistribution(final int limit) {
        double[] result = new double[limit];
        double amount = 1d / limit;
        for(int i = 0; i < limit; i++) {
            result[i] = amount;
        }
        return result;
    }
}

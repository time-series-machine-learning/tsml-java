package utilities;

import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;

public class ArrayUtilities {
    private ArrayUtilities() {}

    public static double[] oneHot(int length, int index) {
        final double[] array = new double[length];
        array[index] = 1;
        return array;
    }

    public static <A> A extractSingleValueList(List<A> list) {
        if(list.size() != 1) {
            throw new IllegalArgumentException("expected a list with only 1 element");
        }
        return list.get(0);
    }

    public static void addInPlace(double[] src, double[] addend) {
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

    public static double[] normaliseInPlace(double[] array, boolean ignoreZeroSum) {
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

    public static double[] normaliseInPlace(double[] array) {
        return normaliseInPlace(array, false);
    }

    public static double[] normalise(double[] array) {
        double[] copy = new double[array.length];
        System.arraycopy(array, 0, copy, 0, array.length);
        return normaliseInPlace(copy);
    }

    public static double[] normalise(int[] array) {
        double sum = sum(array);
        double[] result = new double[array.length];
        if(sum == 0) {
            throw new IllegalArgumentException("sum of zero");
        }
        for(int i = 0; i < array.length; i++) {
            result[i] = array[i] / sum;
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

    public static List<Double> normaliseInPlace(List<Double> list) {
        double sum = sum(list);
        if(sum == 0) {
            throw new IllegalArgumentException("sum zero");
        }
        return list.stream().map(element -> element / sum).collect(Collectors.toList());
    }

    public static List<Double> normalise(Iterable<Double> iterable) {
        List<Double> list = drain(iterable);
        return normaliseInPlace(list);
    }

    public static double[] primitiveIntToDouble(int[] array) {
        double[] doubles = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            doubles[i] = array[i];
        }
        return doubles;
    }

    public static double[] normaliseInPlace(int[] array) {
        return normaliseInPlace(primitiveIntToDouble(array));
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

    public static <A> List<A> flatten(Collection<List<A>> collection) {
        List<A> list = new ArrayList<>();
        for(List<A> subList : collection) {
            list.addAll(subList);
        }
        return list;
    }

    public static int argMax(double[] array) {
        int index = 0;
        double max = array[index];
        for(int i = 1; i < array.length; i++) {
            double value = array[i];
            if(value >= max) {
                if(value > max) {
                    max = value;
                    index = i;
                }
            }
        }
        return index;
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

    public static <A extends Comparable<A>> List<A> uniqueCopy(List<A> values) {
        return unique(new ArrayList<>(values));
    }

    public static <A> List<A> uniqueCopy(List<A> values, Comparator<A> comparator) {
        return unique(new ArrayList<>(values), comparator);
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

    public static int numPermutations(int[] binSizes) {
        return numPermutations(primitiveArrayToList(binSizes));
    }

    public static List<Integer> primitiveArrayToList(int[] values) {
        List<Integer> list = new ArrayList<>();
        for(int i = 0; i < values.length; i++) {
            list.add(values[i]);
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

    public static <A> List<Integer> bestIndices(List<? extends A> list, Comparator<? super A> comparator) {
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

    public static <A> A randomChoice(List<? extends A> list, Random random) {
        return list.get(random.nextInt(list.size()));
    }

    public static <A> int bestIndex(List<? extends A> list, Comparator<? super A> comparator, Random random) {
        List<Integer> indices = bestIndices(list, comparator);
        return randomChoice(indices, random);
    }

    public static <A> int bestIndex(List<? extends A> list, Comparator<? super A> comparator) {
        List<Integer> indices = bestIndices(list, comparator);
        return indices.get(0);
    }


    public static <A extends Comparable<A>> int bestIndex(List<? extends A> list) {
        List<Integer> indices = bestIndices(list, Comparator.naturalOrder());
        return indices.get(0);
    }

    public static <A extends Comparable<? super B>, B extends A> int bestIndex(List<? extends B> list, Random random) {
        return bestIndex(list, Comparable::compareTo, random);
    }

    public static Object[] boxObj(int[] array) {
        Object[] boxed = new Integer[array.length];
        for(int i = 0; i < array.length; i++) {
            boxed[i] = array[i];
        }
        return boxed;
    }

    public static Object[] boxObj(double[] array) {
        Object[] boxed = new Double[array.length];
        for(int i = 0; i < array.length; i++) {
            boxed[i] = array[i];
        }
        return boxed;
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

    public static <A extends Comparable<? super B>, B extends A> A best(final List<? extends B> list, final Random random) {
        return best(list, random, Comparator.naturalOrder());
    }

    public static <A> A best(final List<? extends A> list, final Random random, final Comparator<A> comparator) {
        int bestIndex = bestIndex(list, comparator, random);
        return list.get(bestIndex);
    }

    public static <A> A best(final List<? extends A> list, final Comparator<A> comparator) {
        List<Integer> bestIndices = bestIndices(list, comparator);
        return list.get(bestIndices.get(0));
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


    public static int deepSize(Collection<?> collection) {
        int size = 0;
        for(Object object : collection) {
            if(object instanceof Collection) {
                size += deepSize((Collection<?>) object);
            } else {
                size++;
            }
        }
        return size;
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
        double sum = 0;
        for(int i = 0; i < limit; i++) {
            result[i] = amount;
            sum += amount;
        }
        return result;
    }
}

package utilities;

import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class ArrayUtilities {
    private ArrayUtilities() {}

    public static void add(double[] a, double[] b) {
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

    public static void normalise(double[] array) {
        double sum = sum(array);
        if(sum == 0) {
            throw new IllegalArgumentException("sum of zero");
        }
        for(int i = 0; i < array.length; i++) {
            array[i] /= sum;
        }
    }

    public static void multiply(double[] array, double multiplier) {
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
}

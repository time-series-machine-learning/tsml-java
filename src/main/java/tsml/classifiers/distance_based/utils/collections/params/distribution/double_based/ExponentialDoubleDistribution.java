package tsml.classifiers.distance_based.utils.collections.params.distribution.double_based;

import tsml.classifiers.distance_based.utils.collections.params.distribution.NumericDistribution;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Random;

public class ExponentialDoubleDistribution extends DoubleDistribution {
    public ExponentialDoubleDistribution(final Double min, final Double max) {
        super(min, max);
    }

//    @Override public Double sample() {
//        double lambda = 1;
//        double max = getMax();
//        double min = getMin();
//        double a = Math.exp(-lambda);
//        final Random random = getRandom();
//
//        double u = a * random.nextDouble();
//        double v;
//        if(u == 0) {
//            v = 0 / -lambda;
//        } else {
//            v = Math.log(u) / -lambda;
//        }

//        v = Math.min(max, v);
//        v = Math.max(v, min); // for imprecision overflow outside the bounds
//
//        return v;

//        double max = getMax();
//        double min = getMin();
//        max = Math.exp(-max * lambda);
//        min = Math.exp(-min * lambda);
//        final Random random = getRandom();
//
//        double u = min + (max - min) * random.nextDouble();
//        double v;
//        if(u == 0) {
//            v = 0 / -lambda;
//        } else {
//            v = Math.log(u) / -lambda;
//        }
//
//        v = Math.min(max, v);
//        v = Math.max(v, min); // for imprecision overflow outside the bounds
//
//        return v;
//    }

    public static void main(String[] args) throws IOException {
//        System.out.println(Math.exp(1));
//        System.out.println(Math.exp(0));
//        System.out.println(Math.log(Math.exp(0)));
//        System.out.println(Math.log(Math.exp(1)));
//        System.out.println(Math.log(1));
//        System.out.println();
//        System.setOut(new PrintStream("hello.txt"));
        final ExponentialDoubleDistribution distribution = new ExponentialDoubleDistribution(500d, 505d);
        distribution.setRandom(new Random(0));
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < 1000; i++) {
            double v = distribution.sample();
            if(v < min) {
                min = v;
//                System.out.println("min: " + min);
            }
            if(v > max) {
                max = v;
//                System.out.println("max: " + max);
            }
            System.out.println(v);
        }
    }

    @Override public Double sample() {
        final Random random = getRandom();
        final double u = random.nextDouble();
        final double v = Math.exp(9.21 * u) * 0.0001;
        return v;
    }
}

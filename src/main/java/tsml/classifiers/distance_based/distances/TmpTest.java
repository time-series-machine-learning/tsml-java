package tsml.classifiers.distance_based.distances;

import java.util.Random;
import tsml.classifiers.distance_based.utils.stats.SummaryStat;

public class TmpTest {

    public static void main(String[] args) {
        SummaryStat s1 = new SummaryStat();
        SummaryStat s2 = new SummaryStat();
        long time;
        Random random = new Random(0);
        final int count = 10000000;
        final int repeats = 100;
        for(int i = 0; i < repeats; i++) {
            time = System.nanoTime();
            for(int j = 0; j < count; j++) {
                final double v1 = random.nextDouble();
                final double v2 = random.nextDouble();
                Math.pow(v1 - v2, 2);
            }
            s1.add(System.nanoTime() - time);
            time = System.nanoTime();
            for(int j = 0; j < count; j++) {
                final double v1 = random.nextDouble();
                final double v2 = random.nextDouble();
                t(v1 - v2, 2);
            }
            s2.add(System.nanoTime() - time);
        }
        System.out.println(s1.getMean());
        System.out.println(s2.getMean());
        System.out.println(s1.getPopulationVariance());
        System.out.println(s2.getPopulationVariance());
//        Random random = new Random(0);
//        while(true) {
//            t(random.nextDouble(), random.nextDouble());
//        }
    }

    public static double t(double a, double b) {
//        final double r1 = Math.pow(a - b, 2);
        final double r2 = (a - b) * (a - b);
//        if(r1 != r2) {
//            System.out.println("oops");
//        }
        return r2;
    }
}
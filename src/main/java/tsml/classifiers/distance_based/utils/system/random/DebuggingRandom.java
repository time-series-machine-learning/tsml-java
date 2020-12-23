package tsml.classifiers.distance_based.utils.system.random;

import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

/**
 * Purpose: print off every single random call.
 * <p>
 * Contributors: goastler
 */
public class DebuggingRandom extends Random {

    public DebuggingRandom() {
    }

    public DebuggingRandom(final long l) {
        super(l);
    }

    private int i = 0;
    
    @Override
    public double nextDouble() {
        double v = super.nextDouble();
        System.out.println(i++ + ": nextDouble: " + v);
        return v;
    }

    @Override
    public int nextInt() {
        System.out.println(i++ + ": nextInt()");
        return super.nextInt();
    }

    @Override
    public int nextInt(final int i) {
        int v = super.nextInt(i);
        System.out.println(this.i++ + ": nextInt(" + i + "): " + v);
        return v;
    }

    @Override
    public synchronized void setSeed(final long l) {
        System.out.println(i++ + ": setSeed(" + l + ")");
        super.setSeed(l);
    }

    @Override
    public void nextBytes(final byte[] bytes) {
        System.out.println(i++ + ": nextBytes");
        super.nextBytes(bytes);
    }

    @Override
    public long nextLong() {
        System.out.println(i++ + ": nextLong()");
        return super.nextLong();
    }

    @Override
    public boolean nextBoolean() {
        System.out.println(i++ + ": nextLong()");
        return super.nextBoolean();
    }

    @Override
    public float nextFloat() {
        System.out.println(i++ + ": nextFloat()");
        return super.nextFloat();
    }

    @Override
    public synchronized double nextGaussian() {
        System.out.println(i++ + ": nextGaussian()");
        return super.nextGaussian();
    }

}

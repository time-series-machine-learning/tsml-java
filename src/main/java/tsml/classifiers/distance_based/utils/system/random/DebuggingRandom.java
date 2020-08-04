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

    @Override
    public double nextDouble() {
        double v = super.nextDouble();
        System.out.println("nextDouble: " + v);
        return v;
    }

    @Override
    public int nextInt() {
        System.out.println("nextInt()");
        return super.nextInt();
    }

    @Override
    public int nextInt(final int i) {
        int v = super.nextInt(i);
        System.out.println("nextInt(" + i + "): " + v);
        return v;
    }

    @Override
    public synchronized void setSeed(final long l) {
        System.out.println("setSeed(" + l + ")");
        super.setSeed(l);
    }

    @Override
    protected int next(final int i) {
//        System.out.println("next(" + i + ")");
        return super.next(i);
    }

    @Override
    public void nextBytes(final byte[] bytes) {
        System.out.println("nextBytes");
        super.nextBytes(bytes);
    }

    @Override
    public long nextLong() {
        System.out.println("nextLong()");
        return super.nextLong();
    }

    @Override
    public boolean nextBoolean() {
        System.out.println("nextLong()");
        return super.nextBoolean();
    }

    @Override
    public float nextFloat() {
        System.out.println("nextFloat()");
        return super.nextFloat();
    }

    @Override
    public synchronized double nextGaussian() {
        System.out.println("nextGaussian()");
        return super.nextGaussian();
    }

    @Override
    public IntStream ints(final long l) {
        System.out.println("ints(" + l + ")");
        return super.ints(l);
    }

    @Override
    public IntStream ints() {
        System.out.println("ints()");
        return super.ints();
    }

    @Override
    public IntStream ints(final long l, final int i, final int i1) {
        System.out.println("ints(" + l + "," + i + "," + i1 + ")");
        return super.ints(l, i, i1);
    }

    @Override
    public IntStream ints(final int i, final int i1) {
        System.out.println("ints(" + i + "," + i1 + ")");
        return super.ints(i, i1);
    }

    @Override
    public LongStream longs(final long l) {
        System.out.println("longs(" + l + ")");
        return super.longs(l);
    }

    @Override
    public LongStream longs() {
        System.out.println("longs()");
        return super.longs();
    }

    @Override
    public LongStream longs(final long l, final long l1, final long l2) {
        System.out.println("longs(" + l + "," + l1 + "," + l2 + ")");
        return super.longs(l, l1, l2);
    }

    @Override
    public LongStream longs(final long l, final long l1) {
        System.out.println("longs(" + l + "," + l1 + ")");
        return super.longs(l, l1);
    }

    @Override
    public DoubleStream doubles(final long l) {
        System.out.println("doubles(" + l + ")");
        return super.doubles(l);
    }

    @Override
    public DoubleStream doubles() {
        System.out.println("doubles()");
        return super.doubles();
    }

    @Override
    public DoubleStream doubles(final long l, final double v, final double v1) {
        System.out.println("ints(" + v + "," + v + "," + v1 + ")");
        return super.doubles(l, v, v1);
    }

    @Override
    public DoubleStream doubles(final double v, final double v1) {
        System.out.println("ints(" + v + "," + v1 + ")");
        return super.doubles(v, v1);
    }

}

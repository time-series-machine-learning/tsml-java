package tsml.classifiers.distance_based.utils.random;

import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

/**
 * Purpose: // todo - docs - type the purpose of the code here
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
        super.setSeed(l);
    }

    @Override
    protected int next(final int i) {
        return super.next(i);
    }

    @Override
    public void nextBytes(final byte[] bytes) {
        super.nextBytes(bytes);
    }

    @Override
    public long nextLong() {
        return super.nextLong();
    }

    @Override
    public boolean nextBoolean() {
        return super.nextBoolean();
    }

    @Override
    public float nextFloat() {
        return super.nextFloat();
    }

    @Override
    public synchronized double nextGaussian() {
        return super.nextGaussian();
    }

    @Override
    public IntStream ints(final long l) {
        return super.ints(l);
    }

    @Override
    public IntStream ints() {
        return super.ints();
    }

    @Override
    public IntStream ints(final long l, final int i, final int i1) {
        return super.ints(l, i, i1);
    }

    @Override
    public IntStream ints(final int i, final int i1) {
        return super.ints(i, i1);
    }

    @Override
    public LongStream longs(final long l) {
        return super.longs(l);
    }

    @Override
    public LongStream longs() {
        return super.longs();
    }

    @Override
    public LongStream longs(final long l, final long l1, final long l2) {
        return super.longs(l, l1, l2);
    }

    @Override
    public LongStream longs(final long l, final long l1) {
        return super.longs(l, l1);
    }

    @Override
    public DoubleStream doubles(final long l) {
        return super.doubles(l);
    }

    @Override
    public DoubleStream doubles() {
        return super.doubles();
    }

    @Override
    public DoubleStream doubles(final long l, final double v, final double v1) {
        return super.doubles(l, v, v1);
    }

    @Override
    public DoubleStream doubles(final double v, final double v1) {
        return super.doubles(v, v1);
    }

    @Override
    public int hashCode() {
        return super.hashCode();
    }

    @Override
    public boolean equals(final Object o) {
        return super.equals(o);
    }

    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();
    }

    @Override
    public String toString() {
        return super.toString();
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
    }
}

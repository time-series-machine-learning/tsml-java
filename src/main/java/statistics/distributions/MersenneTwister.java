package statistics.distributions;


import java.io.*;

/**
 * Mersenne Twister and MersenneTwisterFast:
 * <P>
 * <b>MersenneTwister</b> is a drop-in subclass replacement
 * for java.util.Random.  It is properly synchronized and
 * can be used in a multithreaded environment.
 *
 * <p><b>MersenneTwisterFast</b> is not a subclass of java.util.Random.  It has
 * the same public methods as Random does, however, and it is
 * algorithmically identical to MersenneTwister.  MersenneTwisterFast
 * has hard-code inlined all of its methods directly, and made all of them
 * final (well, the ones of consequence anyway).  Further, these
 * methods are <i>not</i> synchronized, so the same MersenneTwisterFast
 * instance cannot be shared by multiple threads.  But all this helps
 * MersenneTwisterFast achieve over twice the speed of MersenneTwister.
 *
 * <p><b>About the Mersenne Twister. </b>
 * This is a Java version of the C-program for MT19937: Integer version.
 * next(32) generates one pseudorandom unsigned integer (32bit)

 * which is uniformly distributed among 0 to 2^32-1  for each
 * call.  next(int bits) >>>'s by (32-bits) to get a value ranging
 * between 0 and 2^bits-1 long inclusive; hope that's correct.
 * setSeed(seed) set initial values to the working area
 * of 624 words. For setSeed(seed), seed is any 32-bit integer
 * <b>except for 0</b>.
 *
 * <p>Orignally Coded by Takuji Nishimura, considering the suggestions by
 * Topher Cooper and Marc Rieffel in July-Aug. 1997.
 * More information can be found
 * <A HREF="http://www.math.keio.ac.jp/matumoto/emt.html">
 * here. </a>

 * <P>
 * Translated to Java by Michael Lecuyer January 30, 1999
 * Copyright (C) 1999 Michael Lecuyer
 * <P>
 * This library is free software; you can redistribute it and or

 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later
 * version.
 * This library is distributed in the hope that it will be useful,

 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Library General Public License for more details.
 * You should have received a copy of the GNU Library General
 * Public License along with this library; if not, write to the
 * Free Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 * <P>
 * Makoto Matsumoto and Takuji Nishimura, the original authors
 * ask "When you use this, send an email to: matumoto@math.keio.ac.jp
 * with an appropriate reference to your work"  You might also point
 * out this was a translation.
 * <P>
 * <b>Reference. </b>
 * M. Matsumoto and T. Nishimura,
 * "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform
 * Pseudo-Random Number Generator",
 * <i>ACM Transactions on Modeling and Computer Simulation,</i>
 * Vol. 8, No. 1, January 1998, pp 3--30.
 *
 * <p><b>About this version. </b>  This is a modification of the
 * <a href="http://www.theorem.com/java/index.htm#Mersenne">original
 * code</a> made to conform to proper java.util.Random format by
 * <a href="http://www.cs.umd.edu/users/seanl/">Sean Luke,</a>
 * August 7, 1999.
 *
 * <p><b>Bug Fixes. </b>This implementation implements the bug fixes made
 * in Java 1.2's version of Random, which means it can be used with
 * earlier versions of Java.  See
 * <a href="http://www.javasoft.com/products/jdk/1.2/docs/api/java/util/Random.html">
 * the JDK 1.2 java.util.Random documentation</a> for further documentation
 * on the random-number generation contracts made.  Additionally, there's
 * an undocumented bug in the JDK java.util.Random.nextBytes() method,
 * which this code fixes.
 *
 * <p><b>Important Note. </b> Just like java.util.Random, this
 * generator accepts a long seed but doesn't use all of it.  java.util.Random
 * uses 48 bits.  The Mersenne Twister instead uses 32 bits (int size).
 * So it's best if your seed does not exceed the int range.
 */

public class MersenneTwister extends java.util.Random implements Serializable
    {
    // Period parameters
    private static final int N = 624;
    private static final int M = 397;
    private static final int MATRIX_A = 0x9908b0df;   //    private static final * constant vector a
    private static final int UPPER_MASK = 0x80000000; // most significant w-r bits
    private static final int LOWER_MASK = 0x7fffffff; // least significant r bits


    // Tempering parameters

    private static final int TEMPERING_MASK_B = 0x9d2c5680;
    private static final int TEMPERING_MASK_C = 0xefc60000;

    // #define TEMPERING_SHIFT_U(y)  (y >>> 11)
    // #define TEMPERING_SHIFT_S(y)  (y << 7)
    // #define TEMPERING_SHIFT_T(y)  (y << 15)
    // #define TEMPERING_SHIFT_L(y)  (y >>> 18)

    private int mt[]; // the array for the state vector
    private int mti; // mti==N+1 means mt[N] is not initialized
    private int mag01[];

    // a good initial seed (of int size, though stored in a long)
    private static final long GOOD_SEED = 4357;


    /**
     * Constructor using the default seed.
     */
    public MersenneTwister()
        {
        super(GOOD_SEED);
        setSeed(GOOD_SEED);
        }

    /**
     * Constructor using a given seed.  Though you pass this seed in
     * as a long, it's best to make sure it's actually an integer.
     *
     * @param seed generator starting number, often the time of day.
     */
    public MersenneTwister(long seed)
        {
        super(seed);    /* just in case */
        setSeed(seed);
        }

    /**
     * Initalize the pseudo random number generator.
     * The Mersenne Twister only uses an integer for its seed;
     * It's best that you don't pass in a long that's bigger
     * than an int.
     *
     * Note that for very old versions of jdk (like 1.0.2),
     * setSeed will not properly reset the gaussian mechanism,
     * so nextGaussian() may return <i>one</i> more extra
     * gaussian drawn from the old seed rather than the new one.
     *
     * @param seed from constructor
     *
     */

    synchronized public void setSeed(long seed)
        {

        // this lets java.util.Random clear its nextNextGaussian field
        // Note this is broken in older jdks like 1.0.2. -- nextNextGaussian
        // will not be cleared so the very next gaussian you get *may* be drawn
        // from the old seed's generation.

        super.setSeed(seed);

        mt = new int[N];

        // setting initial seeds to mt[N] using
        // the generator Line 25 of Table 1 in
        // [KNUTH 1981, The Art of Computer Programming
        //    Vol. 2 (2nd Ed.), pp102]

        // the 0xffffffff is commented out because in Java
        // ints are always 32 bits; hence i & 0xffffffff == i

        mt[0]= ((int)seed); // & 0xffffffff;

        for (mti = 1; mti < N; mti++)
            mt[mti] = (69069 * mt[mti-1]); //& 0xffffffff;

        // mag01[x] = x * MATRIX_A  for x=0,1
        mag01 = new int[2];
        mag01[0] = 0x0;
        mag01[1] = MATRIX_A;
        }

    /**
     * Returns an integer with <i>bits</i> bits filled with a random number.
     */
    synchronized protected int next(int bits)
        {
        int y;

        if (mti >= N)   // generate N words at one time
            {
            int kk;

            for (kk = 0; kk < N - M; kk++)
                {
                y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
                mt[kk] = mt[kk+M] ^ (y >>> 1) ^ mag01[y & 0x1];
                }
            for (; kk < N-1; kk++)
                {
                y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
                mt[kk] = mt[kk+(M-N)] ^ (y >>> 1) ^ mag01[y & 0x1];
                }
            y = (mt[N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
            mt[N-1] = mt[M-1] ^ (y >>> 1) ^ mag01[y & 0x1];

            mti = 0;
            }

        y = mt[mti++];
        y ^= y >>> 11;                          // TEMPERING_SHIFT_U(y)
        y ^= (y << 7) & TEMPERING_MASK_B;       // TEMPERING_SHIFT_S(y)
        y ^= (y << 15) & TEMPERING_MASK_C;      // TEMPERING_SHIFT_T(y)
        y ^= (y >>> 18);                        // TEMPERING_SHIFT_L(y)

        return y >>> (32 - bits);    // hope that's right!
        }


    /* If you've got a truly old version of Java, you can omit these
       two next methods. */

    private synchronized void writeObject(ObjectOutputStream out)
            throws IOException
        {
        // just so we're synchronized.
        out.defaultWriteObject();
        }

    private synchronized void readObject (ObjectInputStream in)
            throws IOException, ClassNotFoundException
        {
        // just so we're synchronized.
        in.defaultReadObject();
        }

    /** This method is missing from jdk 1.0.x and below.  JDK 1.1
        includes this for us, but what the heck.*/
    public boolean nextBoolean() {return next(1) != 0;}

    /** This method is missing from JDK 1.1 and below.  JDK 1.2
        includes this for us, but what the heck. */

    public int nextInt(int n) {
        if (n<=0)
            throw new IllegalArgumentException("n must be positive");

        if ((n & -n) == n)  // i.e., n is a power of 2
            return (int)((n * (long)next(31)) >> 31);

        int bits, val;
        do {
            bits = next(31);
            val = bits % n;
        } while(bits - val + (n-1) < 0);
        return val;
    }

    /** A bug fix for versions of JDK 1.1 and below.  JDK 1.2 fixes
        this for us, but what the heck. */
    public double nextDouble()
        {
        return (((long)next(26) << 27) + next(27))
            / (double)(1L << 53);
        }

    /** A bug fix for versions of JDK 1.1 and below.  JDK 1.2 fixes
        this for us, but what the heck. */


    public float nextFloat()
        {
        return next(24) / ((float)(1 << 24));
        }

    /** A bug fix for all versions of the JDK.  The JDK appears to
        use all four bytes in an integer as independent byte values!
        Totally wrong. I've submitted a bug report. */

    public void nextBytes(byte[] bytes)
        {
        for (int x=0;x<bytes.length;x++)
            bytes[x] = (byte)next(8);
        }

    /** For completeness' sake, though it's not in java.util.Random.  */

    public char nextChar()
        {
        // chars are 16-bit UniCode values
        return (char)(next(16));
        }

    /** For completeness' sake, though it's not in java.util.Random. */

    public short nextShort()

        {
        return (short)(next(16));
        }

    /** For completeness' sake, though it's not in java.util.Random.  */

    public byte nextByte()
        {
        return (byte)(next(8));
        }



    /**
     * Tests the code.
     */
    public static void main(String args[])
        {
        int j;

        MersenneTwister r;

        // UNCOMMENT THIS TO TEST FOR PROPER GAUSSIAN STATE INITIALIZATION

        /*
        System.out.println("If the gaussian state is properly initialized when setSeed() is called,\nthen #1 != #2, but #1 == #3\nIt's known that java 1.0.2 doesn't do gaussian initialization right,\nso setSeed() may result in one last gaussian drawn from the *previous* seed.");
        r = new MersenneTwister(1);
        r.nextGaussian();  // loads the later gaussian into the state
        System.out.println("1:  " + r.nextGaussian());
        r = new MersenneTwister(1);
        r.nextGaussian(); // loads the later gaussian into the state
        r.setSeed(1);     // should reset the gaussian state
        System.out.println("2:  " + r.nextGaussian());
        System.out.println("3:  " + r.nextGaussian());
        */


        // UNCOMMENT THIS TO TEST FOR CORRECTNESS
        // COMPARE WITH http://www.math.keio.ac.jp/~nisimura/random/int/mt19937int.out

        /*
        r = new MersenneTwister(4357);
        System.out.println("Output of MersenneTwister.java");
        for (j=0;j<1000;j++)
            {
            // first, convert the int from signed to "unsigned"
            long l = (long)r.nextInt();
            if (l < 0 ) l += 4294967296L;  // max int value
            String s = String.valueOf(l);
            while(s.length() < 10) s = " " + s;  // buffer
            System.out.print(s + " ");
            if (j%8==7) System.out.println();
            }
        */


        // UNCOMMENT THIS TO TEST FOR SPEED

        /*
        r = new MersenneTwister();
        System.out.println("\nTime to test grabbing 10000000 ints");
        long ms = System.currentTimeMillis();
        int xx=0;
        for (j = 0; j < 10000000; j++)
            xx += r.nextInt();
        System.out.println("Mersenne Twister: " + (System.currentTimeMillis()-ms + "          Ignore this: " + xx));

        Random rr = new Random(1);
        xx = 0;
        ms = System.currentTimeMillis();
        for (j = 0; j < 10000000; j++)
            xx += rr.nextInt();
        System.out.println("java.util.Random: " + (System.currentTimeMillis()-ms + "          Ignore this: " + xx));
        */


        // UNCOMMENT THIS TO DO TEST DIFFERENT TYPE OUTPUTS
        // THIS CAN BE USED TO COMPARE THE DIFFERENCE BETWEEN
        // MersenneTwisterFast.java AND MersenneTwister.java

        /*
        System.out.println("\nGrab the first 1000 booleans");
        r = new MersenneTwister();
        for (j = 0; j < 1000; j++)
            {
            System.out.print(r.nextBoolean() + " ");
            if (j%8==7) System.out.println();
            }
        if (!(j%8==7)) System.out.println();

        byte[] bytes = new byte[1000];
        System.out.println("\nGrab the first 1000 bytes using nextBytes");
        r = new MersenneTwister();
        r.nextBytes(bytes);
        for (j = 0; j < 1000; j++)
            {
            System.out.print(bytes[j] + " ");
            if (j%16==15) System.out.println();
            }
        if (!(j%16==15)) System.out.println();

        byte b;
        System.out.println("\nGrab the first 1000 bytes -- must be same as nextBytes");
        r = new MersenneTwister();
        for (j = 0; j < 1000; j++)
            {
            System.out.print((b = r.nextByte()) + " ");
            if (b!=bytes[j]) System.out.print("BAD ");
            if (j%16==15) System.out.println();
            }
        if (!(j%16==15)) System.out.println();

        System.out.println("\nGrab the first 1000 shorts");
        r = new MersenneTwister();
        for (j = 0; j < 1000; j++)
            {
            System.out.print(r.nextShort() + " ");
            if (j%8==7) System.out.println();
            }
        if (!(j%8==7)) System.out.println();


        System.out.println("\nGrab the first 1000 ints");
        r = new MersenneTwister();
        for (j = 0; j < 1000; j++)
            {
            System.out.print(r.nextInt() + " ");
            if (j%4==3) System.out.println();
            }
        if (!(j%4==3)) System.out.println();

        System.out.println("\nGrab the first 1000 ints of different sizes");
        r = new MersenneTwister();
        for (j = 0; j < 1000; j++)
            {
            System.out.print(r.nextInt(j+1) + " ");
            if (j%4==3) System.out.println();
            }
        if (!(j%4==3)) System.out.println();

        System.out.println("\nGrab the first 1000 longs");
        r = new MersenneTwister();
        for (j = 0; j < 1000; j++)
            {
            System.out.print(r.nextLong() + " ");
            if (j%3==2) System.out.println();
            }
        if (!(j%3==2)) System.out.println();

        System.out.println("\nGrab the first 1000 floats");
        r = new MersenneTwister();
        for (j = 0; j < 1000; j++)
            {
            System.out.print(r.nextFloat() + " ");
            if (j%4==3) System.out.println();
            }
        if (!(j%4==3)) System.out.println();

        System.out.println("\nGrab the first 1000 doubles");
        r = new MersenneTwister();
        for (j = 0; j < 1000; j++)
            {
            System.out.print(r.nextDouble() + " ");
            if (j%3==2) System.out.println();
            }
        if (!(j%3==2)) System.out.println();

        System.out.println("\nGrab the first 1000 gaussian doubles");
        r = new MersenneTwister();
        for (j = 0; j < 1000; j++)
            {
            System.out.print(r.nextGaussian() + " ");
            if (j%3==2) System.out.println();
            }
        if (!(j%3==2)) System.out.println();
        */
        }

    }
package tsml.classifiers.distance_based.utils.classifiers;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.io.NotSerializableException;
import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;

import static org.junit.Assert.*;

public class CopierTest {

    private static class Dummy implements Serializable, Copier {
        public int[] a = {3};
        public String[] b = {"hello"};
        public double p = 1.6;
    }

    private static class BigDummy extends Dummy {
        public Long[] c = {-1L};
    }

    private static class NoDeepCopyDummy {
        public int[] a = {4};
        public String[] b = {"goodbye"};
    }

    private static class AdvDummy {
        public int[] a = {99};
        public String[] b = {":)"};
        public double p = 4.4;
    }

    private static class TransientDummy implements Serializable {
        public transient int[] c;

        public TransientDummy() {
            c = new int[]{11}; // this should be invoked when deep copying
        }
    }

    private Dummy dummy;

    @Before
    public void before() {
        dummy = new Dummy();
    }

    @Test
    public void testDeepCopyTransient() {
        final TransientDummy dummy = new TransientDummy();
        dummy.c[0] = 15;
        final TransientDummy copy = CopierUtils.deepCopy(dummy);
        assertNotSame(copy, dummy);
        assertNotSame(copy.c, dummy.c);
        assertEquals(11, copy.c[0]);
    }

    @Test
    public void testShallowCopyString() {
        String a = "hello";
        String b = CopierUtils.shallowCopy(a);
        assertEquals(a, b);
        assertNotSame(a, b);
    }

    @Test
    public void testShallowCopy() {
        final Dummy copy = (Dummy) dummy.shallowCopy();
        assertSame(dummy.a, copy.a);
        assertSame(dummy.b, copy.b);
        assertNotSame(copy, dummy);
    }

    @Test
    public void testDeepCopy() {
        final Dummy copy = (Dummy) dummy.deepCopy();
        assertNotSame(dummy.a, copy.a);
        assertArrayEquals(dummy.a, copy.a);
        assertNotSame(dummy.b, copy.b);
        assertArrayEquals(dummy.b, copy.b);
        assertNotSame(copy, dummy);
    }

    @Test
    public void testInheritanceDeepCopy() {
        final BigDummy copy = new BigDummy();
        CopierUtils.deepCopy(dummy, copy);
        assertNotSame(dummy.a, copy.a);
        assertArrayEquals(dummy.a, copy.a);
        assertNotSame(dummy.b, copy.b);
        assertArrayEquals(dummy.b, copy.b);
        assertNotSame(copy, dummy);
        assertEquals(copy.c[0], new Long(-1L));
    }

    @Test
    public void testInheritanceShallowCopy() {
        final BigDummy copy = new BigDummy();
        CopierUtils.shallowCopy(dummy, copy);
        assertSame(dummy.a, copy.a);
        assertArrayEquals(dummy.a, copy.a);
        assertSame(dummy.b, copy.b);
        assertArrayEquals(dummy.b, copy.b);
        assertNotSame(copy, dummy);
        assertEquals(copy.c[0], new Long(-1L));
    }

    @Test
    public void testPartialDeepCopy() throws IOException {
        final Dummy copy = new BigDummy();
        copy.a[0] = 5;
        CopierUtils.deepCopy(dummy, copy, "b");
        assertNotSame(dummy.a, copy.a);
        assertNotSame(dummy.b, copy.b);
        assertArrayEquals(dummy.b, copy.b);
        assertNotSame(copy, dummy);
    }

    @Test
    public void testCopyByFieldNames() {
        final AdvDummy copy = new AdvDummy();
        CopierUtils.deepCopy(dummy, copy);
        assertNotSame(dummy.a, copy.a);
        assertArrayEquals(dummy.a, copy.a);
        assertNotSame(dummy.b, copy.b);
        assertArrayEquals(dummy.b, copy.b);
        assertNotSame(copy, dummy);
    }

    @Test
    public void testPartialShallowCopy()
            throws InstantiationException, ClassNotFoundException, InvocationTargetException, IOException {
        final Dummy copy = new BigDummy();
        copy.a[0] = 5;
        CopierUtils.shallowCopy(dummy, copy, "b");
        assertNotSame(dummy.a, copy.a);
        assertSame(dummy.b, copy.b);
        assertArrayEquals(dummy.b, copy.b);
        assertNotSame(copy, dummy);
    }

    @Test(expected = IllegalStateException.class)
    public void testDeepCopyNoSerialisable() {
        final NoDeepCopyDummy test = new NoDeepCopyDummy();
        test.a[0] = 5;
        CopierUtils.deepCopy(test);
    }
}

package tsml.classifiers.distance_based.utils.classifiers;

import tsml.classifiers.distance_based.utils.classifiers.checkpointing.BaseCheckpointer;

import java.io.Serializable;

public class Tmp {
    private static class Test implements Serializable, Copier {
        private Test(final long v) {
            this.v = v;
        }

        private long v;
    }

    public static void main(String[] args) throws Exception {
        Test t1 = new Test(3);
        Test t2 = new Test(5);
        BaseCheckpointer a = new BaseCheckpointer(t1);
        BaseCheckpointer b = new BaseCheckpointer(t2);
        Copier.copyFields(a, b);
    }
}

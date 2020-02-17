package tsml.classifiers.distance_based.pf;
/*

purpose: // todo - docs - type the purpose of the code here

created edited by goastler on 17/02/2020
    
*/

import weka.core.Instances;

public class Info {

    public Info() {

    }

    public Info(Info other) {
        throw new UnsupportedOperationException();
    }

    private Instances data;

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }
}

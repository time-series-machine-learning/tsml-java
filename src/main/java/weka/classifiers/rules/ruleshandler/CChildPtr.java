package weka.classifiers.rules.ruleshandler;

public class CChildPtr {

    CFpNode child;
    CChildPtr next;

    public CChildPtr(int k) {

        child = new CFpNode(k);
        next = null;
    }
}

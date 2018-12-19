package weka.classifiers.rules.ruleshandler;

public class CFpNode {

    int itemId;
    CFpNode parent;
    CChildPtr children;
    int local_supp;
    int[] local_suppClass;
    CFpNode next;

    public CFpNode(int k) {

        itemId = 0;
        local_supp = 0;
        local_suppClass = new int[k];
        children = null;
    }
}

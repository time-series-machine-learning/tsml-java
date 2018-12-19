package weka.classifiers.rules.ruleshandler;

public class CFptree {

    CFpNode root;
    int branches;

    public CFptree(int k) {

        root = new CFpNode(k);
    }

    public void setBranches(int j) {
        branches = j;
    }
}

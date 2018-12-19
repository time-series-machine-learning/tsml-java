package weka.classifiers.rules.ruleshandler;

public class CItemEntry {

    int itemId;
    int supp;
    int[] suppClass;
    CItemEntry next;

    public CItemEntry(int k) {
        itemId = 0;
        supp = 0;
        suppClass = new int[k];
        next = null;
    }
}

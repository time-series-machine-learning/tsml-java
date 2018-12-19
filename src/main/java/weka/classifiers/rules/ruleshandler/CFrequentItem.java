package weka.classifiers.rules.ruleshandler;

import java.util.*;


public class CFrequentItem {

    int itemId;
    int supp;
    int[] suppClass;
    CFpNode head;
    int nodeLinkCounter;
    int accorpato;
    LinkedList<Integer> itemAccorpati;

    public CFrequentItem(int k) {

        itemId = 0;
        supp = 0;
        nodeLinkCounter = 0;
        accorpato = 0;
        suppClass = new int[k];
        head = null;
        itemAccorpati = new LinkedList<Integer>();
    }

    public void incCounter(int g) {
        nodeLinkCounter = nodeLinkCounter + g;
    }

}

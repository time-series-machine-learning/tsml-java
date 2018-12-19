package weka.classifiers.rules;

import java.util.*;

public class Transaction {
    // attributes
    private int tid;
    private int cid;
    private int Class_ID;
    private int Num_Items;
    private Item Items[];
    private int Last_Rule;

    // constructor
    public Transaction() {
    }

    public Transaction(int tid, int cid, int Class_ID, int Num_Items, Item Items[]) {
        this.tid = tid;
        this.cid = cid;
        this.Class_ID = Class_ID;
        this.Num_Items = Num_Items;
        this.Items = Items;
        this.Last_Rule = -1;
        Arrays.sort(this.Items);
    }

    // methods
    public int gettid() {
        return tid;
    }

    public int getcid() {
        return cid;
    }

    public Item[] getItems() {
        return Items;
    }

    public int getClass_ID() {
        return Class_ID;
    }


    public int getNum_Items() {
        return Num_Items;
    }

    public void setLast_Rule(int last_rule) {
        this.Last_Rule = last_rule;
    }

    public int getLast_Rule() {
        return Last_Rule;
    }


}


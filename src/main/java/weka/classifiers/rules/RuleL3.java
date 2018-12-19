package weka.classifiers.rules;

import java.io.*;
import java.util.*;
import java.util.ArrayList;
import java.io.IOException;


public class RuleL3 {
    // attributes
    private int Rule_ID;
    private int Class_ID;
    private int Length;
    private int Absolute_Support;
    private float Confidence;
    private Item Items[];
    private int Correct;
    private int Incorrect;
    private int Level;


    // constructors

    public RuleL3() {
    }

    public RuleL3(String rule, int id) throws IOException {
        this.Rule_ID = id;

        String regex = "->";
        String compact_rule= rule.split(regex)[1];


        ArrayList<Double> list = new ArrayList<Double>();
        StringReader str = new StringReader(compact_rule);
        StreamTokenizer stok = new StreamTokenizer(str);
        stok.parseNumbers();
        stok.nextToken();

        while (stok.ttype != StreamTokenizer.TT_EOF) {
            if (stok.ttype == StreamTokenizer.TT_NUMBER)
                list.add( new Double(stok.nval));
            stok.nextToken();
        }

        this.Class_ID = (int)list.get(0).doubleValue();

        this.Length = (int)list.get(3).doubleValue();

        this.Absolute_Support = (int)list.get(1).doubleValue();
        this.Confidence = (float)list.get(2).doubleValue();

        this.Items = new Item[this.Length];
        for (int i = 0; i < this.Length; i++)
            this.Items[i] = new Item((int)list.get(4+i).doubleValue());
        Arrays.sort(this.Items);

        this.Correct = 0;
        this.Incorrect = 0;
        this.Level = 0;
    }

    // methods
    public int getRule_ID() {
        return Rule_ID;
    }

    public int getClass_ID() {
        return Class_ID;
    }

    public int getLength() {
        return Length;
    }

    public int getAbsolute_Support() {
        return Absolute_Support;
    }

    public float getConfidence() {
        return Confidence;
    }

    public Item[] getItems() {
        return Items;
    }

    public int getCorrect() {
        return Correct;
    }

    public int getIncorrect() {
        return Incorrect;
    }

    public int getLevel() {
        return Level;
    }



    public void classifyTrans(Transaction trans) {
        boolean matches = true;
        int i = 0;
        int j = 0;

        while (matches && i<this.Items.length) {
            j=0;
            matches = false;
            while (!matches && j<trans.getItems().length) {
                if (this.Items[i].getValue()==trans.getItems()[j].getValue())
                    matches = true;
                j++;
            }
            i++;
        }

        if (matches) {
            trans.setLast_Rule(Rule_ID);
            if (this.Class_ID == trans.getClass_ID())
                this.Correct++;
            else
                this.Incorrect++;
        }

    }

    public void setLevel() {
        if (this.Correct > 0)
            this.Level = 1;
        else if (this.Correct == 0 && this.Incorrect == 0)
            this.Level = 2;
    }

}


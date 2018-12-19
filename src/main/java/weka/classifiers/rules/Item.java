package weka.classifiers.rules;



public class Item implements Comparable {
    //attrubutes
    private int Value;

    //constructor
    public Item(int Value) {
        this.Value = Value;
    }

    //methods

    public void setValue(int value) { // sets value of the Item
        this.Value = value;
    }

    public int getValue() { // returns value of the Item
        return Value;
    }

    public int compareTo(Object anotherItem) throws ClassCastException {
        if (!(anotherItem instanceof Item))
            throw new ClassCastException("An Item object expected.");
        int anotherItemValue = ((Item) anotherItem).getValue();
        return this.Value - anotherItemValue;
    }
}


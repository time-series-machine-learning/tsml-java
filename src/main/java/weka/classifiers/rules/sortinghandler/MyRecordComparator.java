package weka.classifiers.rules.sortinghandler;

/**
 *  This code is from the book:
 *
 *    Winder, R and Roberts, G (1998) <em>Developing Java
 *    Software</em>, John Wiley & Sons.
 *
 *  It is copyright (c) 1997 Russel Winder and Graham Roberts.
 */

import weka.classifiers.rules.sortinghandler.Comparator ;
import weka.classifiers.rules.sortinghandler.ComparatorParameterErrorException ;

public class MyRecordComparator implements Comparator
{
    public boolean relation(Object a, Object b)
    {
        if (! (a instanceof MyRecord))
        {
            throw new ComparatorParameterErrorException
                ("RecordComparator parameter 1") ;
        }
        if (! (b instanceof MyRecord))
        {
            throw new ComparatorParameterErrorException
                ("RecordComparator parameter 2") ;
        }
        return execute((MyRecord)a, (MyRecord)b) ;
    }

    public static boolean execute(MyRecord a, MyRecord b)
    {
    	boolean value = false;
    	// here I handle record comparation
    	if (a.confidence < b.confidence)
    		value = false; // b precedes a
    	else 
    	{ 
    		if (a.confidence == b.confidence)
    		{
    			if (a.support < b.support)
    				value = false; // b precedes a
    			else
    			{
    				if (a.support == b.support)
    				{
    					if (a.length < b.length)
    						value = false; // b precedes a
    					else
    					{
    						
    						if (a.length == b.length)
    						{
    						} 
    						else
    						{
    							value = true; // a precedes b
    						}
    					}
    				}
    				else
    				{
    					value = true; // a precedes b
    				}
    			}
    		}
    		else 
    		{
    		value = true; // a precedes b
    		}
    		
    	}
    	return value;
    }
}


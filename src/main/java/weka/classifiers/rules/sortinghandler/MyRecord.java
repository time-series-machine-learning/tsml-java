package weka.classifiers.rules.sortinghandler;

/**
 *  This code is from the book:
 *
 *    Winder, R and Roberts, G (1998) <em>Developing Java
 *    Software</em>, John Wiley & Sons.
 *
 *  It is copyright (c) 1997 Russel Winder and Graham Roberts.
 */

import java.util.StringTokenizer;

import weka.classifiers.rules.sortinghandler.Record ;

public class MyRecord implements Record
{
    public MyRecord(String s)
    {
        data = s ;
        /*** my updating ****/
        if (s!= null) 
        {
        // parse the string data extracting confidence, support and length
        StringTokenizer st = new StringTokenizer(data, " ");
	    int counter_tokens = 0;
	    while(st.hasMoreTokens()) {
	    	String str = st.nextToken();
	    	counter_tokens++;
	    	switch (counter_tokens){
	    	case 1: // first token. The unordered rule. Useless!
	    		break;
	    	case 2: // second token: "->" meaningless!
	    		break;
	    	case 3: // third token: class_id. Useless!
	    		break;
	    	case 4: // fourth token: support
	    		support = Integer.parseInt(str);
	    		//System.out.println("sup: "+support);
	    		break;
	    	case 5: // fifth token: confidence
	    		confidence = Double.parseDouble(str);
	    		break;
	    	case 6: // sixth token: rule length
	    		length = Integer.parseInt(str);
	    		//System.out.println("length: "+length);
	    		break;
	    	default: // further tokens: ordered rules' list
	    		break;
	    	} // end switch
	    } // end while
        } // end if
        else
        	System.out.println("record null!!\n");
	    /*** end of my updating ****/
    }
    
    public int key()
    {
        return Integer.parseInt(data) ;
    }

    public String toString()
    {
        return data ;
    }
    
    private String data ;
    /*** my updating ****/
    public long class_id;
    public int support;
    public double confidence;
    public int length;
    public static int MAX_RULE_LENGTH = 200;
    //public int ordered_list[];
    /*** end of my updating ****/
    
}


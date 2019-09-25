/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package weka_extras.classifiers.ensembles;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import utilities.ClassifierTools;

/**
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPETest {
    
    public CAWPETest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
        System.out.println("-----Start CAWPE tests-----");
    }
    
    @AfterClass
    public static void tearDownClass() {
        System.out.println("-----End CAWPE tests-----\n");
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

//    /**
//     * Test of buildCAWPEPaper_AllResultsForFigure3 method, of class CAWPE.
//     */
//    @Test
//    public void testBuildCAWPEPaper_AllResultsForFigure3() throws Exception {
//        System.out.println("buildCAWPEPaper_AllResultsForFigure3");
//        CAWPE.buildCAWPEPaper_AllResultsForFigure3();
//        // TODO review the generated test code and remove the default call to fail.
//        fail("The test case is a prototype.");
//    }

    /**
     * Test of simple results reproduction, of class CAWPE.
     */
    @Test
    public void testReproduction() throws Exception {
        System.out.println("--testReproduction()");
        double expectedAcc = 0.0;
//        double expectedAcc = 0.9650145772594753;
        System.out.println("CAWPE expected accuracy generated 2019_09_25");
        assertTrue(ClassifierTools.testUtils_confirmIPDReproduction(new CAWPE(), expectedAcc));
    }
    
}

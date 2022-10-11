## Contents

[Structure](#structure)  
[Example Solution](#example-solution)  
[Notes and Takeaways](#notes-and-takeaways)

---

## Structure
main.py sets up the GA and has functions to define how the GA should mutate or create new genes. The GA package is simply a framework to call those functions. I'm not convinced this was the best way to go about structuring the project, but I wanted to try to generalize the GA package while still allowing the user to define his/her own methodds...

---

## Example Solution

- Population Size: 1000  
- Minimum Generations: 200  
- Minimum Improvement: 0.01  

<img src="https://github.com/nosv1/seagraves_intro_to_ai/blob/master/HW2_GeneticAlogorithm/GA%201000x150%200o01.png?raw=true">

Alternate views of the solution can be found [here](https://github.com/nosv1/seagraves_intro_to_ai/tree/master/HW2_GeneticAlogorithm/Solution).

## **End Fitness: 16.20** 


Course: CS101A  
Class Time: 12:00 PM - 12:50 PM  
Instructor: Zein el Din  
Room: Royall (201) 50/50 (100.00% full)  

Course: CS101B  
Class Time: 11:00 AM - 11:50 AM  
Instructor: Hare  
Room: Bloch (119) 50/60 (83.33% full)  

Course: CS191A  
Class Time: 01:00 PM - 01:50 PM  
Instructor: Zein el Din  
Room: Bloch (119) 50/60 (83.33% full)  

Course: CS191B  
Class Time: 10:00 AM - 10:50 AM  
Instructor: Zein el Din  
Room: Haag (201) 50/60 (83.33% full)  

Course: CS201  
Class Time: 10:00 AM - 10:50 AM  
Instructor: Hare  
Room: FH (310) 50/108 (46.30% full)  

Course: CS291  
Class Time: 01:00 PM - 01:50 PM  
Instructor: Hare  
Room: Haag (301) 50/75 (66.67% full)  

Course: CS303  
Class Time: 12:00 PM - 12:50 PM  
Instructor: Hare  
Room: Haag (301) 60/75 (80.00% full)  

Course: CS304  
Class Time: 11:00 AM - 11:50 AM  
Instructor: Zein el Din  
Room: FH (216) 25/30 (83.33% full)  

Course: CS394  
Class Time: 01:00 PM - 01:50 PM  
Instructor: Xu  
Room: Katz (003) 20/45 (44.44% full)  

Course: CS449  
Class Time: 12:00 PM - 12:50 PM  
Instructor: Xu  
Room: Haag (201) 60/60 (100.00% full)  

Course: CS451  
Class Time: 02:00 PM - 02:50 PM  
Instructor: Xu  
Room: FH (310) 100/108 (92.59% full)    

</br>

## Checks
same_room_same_time: 0 * -0.5 = -0.0  

room_too_small: 0 * -0.5 = -0.0  
room_3x_too_big: 0 * -0.2 = -0.0  
room_6x_too_big: 0 * -0.4 = -0.0  
room_size_sufficient: 11 * 0.3 = 3.3  

preferred_instructor: 10 * 0.5 = 5.0/5.5  
other_instructor: 1 * 0.2 = 0.2  
other_faculty: 0 * -0.1 = -0.0  

instructor_one_class_one_time: 11 * 0.2 = 2.2/2.2  
instructor_multiple_classes_one_time: 0 * -0.2 = -0.0  
instructor_more_than_4_classes: 0 * -0.5 = -0.0  
instructor_less_than_3_classes: 0 * -0.4 = -0.0  
instructor_consecutive_slots: 8 * 0.5 = 4.0/4.0  
instructor_consecutive_slots_far_away_rooms: 0 * -0.4 = -0.0  

cs_101_4_hour_gap: 0 * 0.5 = 0.0  
cs_101_same_time: 0 * -0.5 = -0.0  
cs_191_4_hour_gap: 0 * 0.5 = 0.0  
cs_191_same_time: 0 * -0.5 = -0.0  
cs_101_191_consecutive: 2 * 0.5 = 1.0/2.0  
sections_consecutive_far_away_rooms: 0 * -0.4 = -0.0  
cs_101_191_one_hour_gap: 2 * 0.25 = 0.5  
cs_101_191_same_time: 0 * -0.25 = -0.0  

---

## Notes and Takeaways

### **Overall**
Overall the project went well. The actual GA part isn't so difficult. The hardest part was validating the fitness function - even writing this now, I'm only so confident it's producing correct values. I've come up with an esitmate fitness of 19 for an optimal schedule, and given the GA is outputting a fitness of 16 or so with some observable penalties and missed opportunities (seen in Solution/Checks.txt), I think it's working...

### **Parallelization**
I decided not to parallelize (after testing)... with a small population (500 or so) there ends up being only a small benefit given the overhead needed to pass the population between threads, and given the calculation of the fitness function isn't difficult, I'm not unhappy with the speeds between generations (2 seconds or so for population of 1000).

### **Fitness Function**
I disagree with some of the weights used for the fitness function. For exmaple, evaluating a solution with 2 classes in the same room at the same time should be a massive penalty, along with an instructor needing to be in two places at the same time, these solutions are impossible and should be weighted as such - which is hopefully a fair opinion.

I did make an executive decision to adjust the check for the 'classes in far away buildings'. The assignment read, "with consecutive time slots, one of the classes is in Bloch or Katz and the other isn't: -0.4". My interpretation is that we're penalizing consecutive classes that aren't in the same remote building. I imagine what was meant was if we go from Bloch to Katz in consecutive classes, then -0.4. This allows consecutive classes to be in a remote building or the quad without penalty.

### **Selection and Mutation**
For the selection, my process was simple, as I imagine the randomness can only be so useful... 
- Evaluate all chromosomes to get a fitness score
- Generate a list of probabilities using softmax based on those fitness scores
- Loop population size randomly selecting chromosomes based on those probabilities
- Loop genes of selected chromosomes and randomly mutate them based on mutation rate

This creates a slightly mutated generation from decent chromosomes. That being said, it was suggested a mutaiton rate of 0.01 was too high, however, with minimal mutattion randomness, the starting mutation rate wound up being pretty decent. When I raised the mutation to 0.05, it would produce worse results, and the same was observed when I decreased it to 0.005.


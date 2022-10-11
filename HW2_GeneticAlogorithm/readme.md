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
- Minimum Generations: 150  
- Minimum Improvement: 0.01  

<img src="https://github.com/nosv1/seagraves_intro_to_ai/blob/master/HW2_GeneticAlogorithm/GA%201000x150%200o01.png?raw=true">

Alternate views of the solution can be found [here](https://github.com/nosv1/seagraves_intro_to_ai/tree/master/HW2_GeneticAlogorithm/Solution).

## **End Fitness: 16.20 / 17.50** 


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

preferred_instructor: 10 * 0.5 = 5.0  
other_instructor: 1 * 0.2 = 0.2  
other_faculty: 0 * -0.1 = -0.0  

instructor_one_class_one_time: 11 * 0.2 = 2.2  
instructor_multiple_classes_one_time: 0 * -0.2 = -0.0  
instructor_more_than_4_classes: 0 * -0.5 = -0.0  
instructor_less_than_3_classes: 0 * -0.4 = -0.0  
instructor_consecutive_slots: 8 * 0.5 = 4.0  
instructor_consecutive_slots_far_away_rooms: 0 * -0.4 = -0.0  

cs_101_5_hour_gap: 0 * 0.5 = 0.0  
cs_101_same_time: 0 * -0.5 = -0.0  
cs_191_5_hour_gap: 0 * 0.5 = 0.0  
cs_191_same_time: 0 * -0.5 = -0.0  
cs_101_191_consecutive: 2 * 0.5 = 1.0
sections_consecutive_far_away_rooms: 0 * -0.4 = -0.0  
cs_101_191_one_hour_gap: 2 * 0.25 = 0.5  
cs_101_191_same_time: 0 * -0.25 = -0.0  

---

## Notes and Takeaways

### **Overall**
Overall the project went well. The actual GA part wasn't so difficult. The hardest part was validating the fitness function - even writing this now, I'm only so confident it's producing correct values. If my math is correct, I beleive the maximum fitness possible is 17.5, but there lies conflicts where I also beleive a perfect solution is not possible (more details below).

11 sufficent rooms: 11 * 0.3 = 3.3  
11 preferred instructors: 11 * 0.5 = 5.5  
11 instructors one class one time: 11 * 0.2 = 2.2  
8 consecutive slots for instructors (2 x 3 + 2): 8 * 0.5 = 4.0  
sections 5 hour gaps: 2 * 0.5 = 1  
sections consecutive: 2 * 0.5 = 1  
sepearate course sections one hour gap: 2 * 0.25 = 0.5  

### **Fitness Function**
I disagree with some of the weights used for the fitness function. For exmaple, evaluating a solution with 2 classes in the same room at the same time should be a massive penalty, along with an instructor needing to be in two places at the same time, these solutions are impossible and should be weighted as such - which is hopefully a fair opinion.

I did make an executive decision to adjust the check for the 'classes in far away buildings'. The assignment read, "with consecutive time slots, one of the classes is in Bloch or Katz and the other isn't: -0.4". My interpretation is that we're penalizing consecutive classes that aren't in the same remote building. I imagine what was meant was if we go from Bloch to Katz in consecutive classes, then -0.4. This allows consecutive classes to be in a remote building or the quad without penalty.

Regarding the max fitness not being possible, there was at least one conflict that was noticeable. The assignment says "The 2 sections of CS101/191 are more than 4 hours apart to get a bonus - I interpreted this as "start time minus start time > 4 hours". There's also a bonus for two different courses being consecutive (CS191 follows CS101 x2 for both sections). Here lies the problem, though. The time slots provided have the first class starting at 10:00 and the last one starting at 15:00. This means a section of CS101 needs to start at 10:00, a section CS191 needs to start at 11:00, then the other CS101 needs to start 5 hours later than the first one, so 10:00 + 5 hours = 15:00. This leaves no room for the other section of CS191 to start at 16:00, meaning there does not exist a solution that will satisfy every constraint :(

### **Selection and Mutation**
For the selection, my process was simple, as I imagine the randomness can only be so useful... 
1. Evaluate all chromosomes to get a fitness score
2. Generate a list of probabilities using softmax based on those fitness scores
3. Loop population size randomly selecting chromosomes based on those probabilities
4. Loop genes of selected chromosomes and randomly mutate them based on mutation rate

This creates a slightly mutated generation from decent chromosomes. That being said, it was suggested a mutaiton rate of 0.01 was too high, however, with minimal mutation randomness, the starting mutation rate wound up being pretty decent. When I raised the mutation to 0.05, it would produce worse results, and the same was observed when I decreased it to 0.005.

### **Parallelization**
I decided not to parallelize (after testing)... with a small population (500 or so) there ends up being only a small benefit given the overhead needed to pass the population between threads, and given the calculation of the fitness function isn't difficult, I'm not unhappy with the speeds between generations (<1 second or so for population of 1000).
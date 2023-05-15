# SCP Classification Model

## A work in progress machine learning project by Daniel Fischer

### What is the goal of this project?

The current goal of this project is to develop a machine learning model that is able use the frequency of each unique word of text articles from [the SCP Wiki](https://scp-wiki.wikidot.com) to determine the nature of the article and correctly classify it. The SCP Wiki is a collaborative internet writing project in which users submit articles outlining anomalous creatures, objects and/or locations, posing as a member of the fictional "SCP Foundation" who seek out and try to contain these anomalies.  

Each article on the site is assigned an 'object class' attribute which represents how difficult the anomaly is to contain. There are four object classes that encomoass the majority of the articles appearing on the SCP Wiki. They are defined as follows:  

- Safe:
  > Safe-class SCPs are anomalies that are easily and safely contained. This is often due to the fact that the Foundation has researched the SCP well enough that containment does not require significant resources or that the anomalies require a specific and conscious activation or trigger. Classifying an SCP as Safe, however, does not mean that handling or activating it does not pose a threat.

- Euclid:
    > Euclid-class SCPs are anomalies that require more resources to contain completely or where containment isn't always reliable. Usually this is because the SCP is insufficiently understood or inherently unpredictable. Euclid is the Object Class with the greatest scope, and it's usually a safe bet that an SCP will be this class if it doesn't easily fall into any of the other standard Object Classes. As a note, any SCP that's autonomous, sentient and/or sapient is generally classified as Euclid, due to the inherent unpredictability of an object that can act or think on its own.

- Keter:
  > Keter-class SCPs are anomalies that are exceedingly difficult to contain consistently or reliably, with containment procedures often being extensive and complex. The Foundation often can't contain these SCPs well due to not having a solid understanding of the anomaly, or lacking the technology to properly contain or counter it. A Keter SCP does not mean the SCP is dangerous, just that it is simply very difficult or costly to contain.
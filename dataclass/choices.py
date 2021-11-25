from enum import Enum

def ChoiceEnum(choices):
    return Enum('choices',{k:k for k in choices})

PIVOTTABLE_CHOICES = ChoiceEnum(['all','train','test'])
BODY_CHOICES = ChoiceEnum(['meta','base','tree','extra'])
TUPLETRIPPLE_CHOICES = ChoiceEnum(['basic','tuple','tripple'])

#%%

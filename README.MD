Et projekt i forbindelse med deep learning kurset på DTU.

VÆR OBS PÅ AT DATA IKKE KOMMER MED OP. EN RODET OMGANG AF Resumé

Kristians idéer

Lige nu med data kommer vi til at finde for mange anomalies, fx hvaler, biler osv.

Instantaneous anamoli detection, kan måske også give en simpel klasse må også gerne gøres efter.

Vi må gerne bare køre CV på det.


Start med anamalies detecetion instance segmentation. DET HER er essensen. Gerne så hurtigt som muligt, kig på noget parralization.

derefter clustering modell og clustering. måske en identifikations model med det samme.

Man kan køre en identifikationsmodel som trin to direkte efter anomlies detection.

Kristian antager at objekterne indenfor clusterene er samme objekt. Så det kan godt være lidt en "manuel identifikation".

Korte af det lange

Find alle objekter, eller så mange som muligt.

Se om vi kan clustere dem

Lav noget self identifikation hvis muligt.

Kig på HDBSCAN.
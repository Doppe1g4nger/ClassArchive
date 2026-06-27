# ClassArchive
This WIP repo archives class files (programming assignments, independent studies, etc.) from undergraduate study at the University of Kentucky

Note this repo does not include a license meaning I withhold ALL rights to the included code, docs, whatever. If you plagiarize it for an assignment, you're the one with your neck on the line.

## Projects

### rfdes — component-level discrete-event RF system framework

A Python framework for modeling an entire RF system at the component level
inside a discrete-event simulation. It runs as a guest inside an external
simulator that owns the event queue and delivers IQ buffers via a `signalRX`
event; components are composed dynamically (`subscribe` / `>>`), and each
component's processing and the data flow between components are modeled as
events on that queue, each with a per-component processing delay. See
[`src/rfdes/README.md`](src/rfdes/README.md).

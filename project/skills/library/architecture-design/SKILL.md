---
name: architecture-design
description: Guides the design of software architecture, focusing on scalability, maintainability, and pattern selection. Use when starting a new project or refactoring a system.
---

# Architecture Design Workflow

This skill assists in designing robust software architectures. It encourages a structured approach starting from requirements to component selection.

## Stage 1: Requirements Gathering

**Goal:** Define functional and non-functional requirements.

### Questions
1. **Functional:** What must the system do? (User stories)
2. **Scale:** Expected load (RPS, data volume)?
3. **Availability:** SLA requirements (99.9% vs 99.99%)?
4. **Constraints:** Budget, existing tech stack, timeline?

## Stage 2: Pattern Selection

**Goal:** Choose the right architectural pattern.

### Common Patterns
- **Monolithic:** Good for simple, low-scale apps.
- **Microservices:** Good for complex, high-scale, multi-team apps.
- **Event-Driven:** Good for decoupling and high throughput.
- **Layered (Clean/Onion):** Good for business logic separation.

**Decision Matrix:** Compare patterns against requirements (e.g., Microservices add complexity but enable independent scaling).

## Stage 3: Component Design

**Goal:** Define key components and their interactions.

### Components
- **Frontend/Client:** Web, Mobile, CLI.
- **API Gateway:** Entry point, auth, rate limiting.
- **Services:** Core business logic units.
- **Storage:** SQL (Relational), NoSQL (Document/Key-Value), Blob Storage.
- **Communication:** REST, gRPC, Message Queues (Kafka/RabbitMQ).

## Stage 4: Data Flow & Interface Definition

**Goal:** Map how data moves through the system.

### Deliverables
- **Data Flow Diagram:** User -> API -> Service -> DB.
- **API Spec:** High-level endpoints (e.g., `POST /users`, `GET /orders/{id}`).
- **Data Model:** Entity Relationship Diagram (ERD).

## Stage 5: Review & Refinement

**Goal:** Identify bottlenecks and failure points.

### Analysis
- **Single Points of Failure (SPOF):** What happens if the DB goes down?
- **Scalability:** Can we add more nodes?
- **Security:** Authentication, Authorization, Encryption.

## Execution Instructions
- Start high-level, then drill down.
- Always justify technology choices based on requirements (no "Resume Driven Development").
- Output a textual description or Mermaid diagram code.

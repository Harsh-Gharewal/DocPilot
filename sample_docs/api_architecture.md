# Internal API & Microservice Architecture

## Authentication & Authorization
All incoming requests to the internal cluster must pass through the API Gateway. The gateway validates JSON Web Tokens (JWT) using the `auth.jwt_secret` configuration key. Tokens must contain valid claims including `user_id`, `role`, and `tenant_id`.

## Database Connection Pooling
The service interacts with PostgreSQL using an asynchronous connection pool managed by SQLAlchemy and asyncpg. The pool size is set to a maximum of 20 connections per pod with a 10-second timeout. Any unhandled database connection exhaustion emits error code `ERR_DB_POOL_EXHAUSTED`.

## Rate Limiting Policy
Public client IPs are restricted to 100 requests per minute. Internal service-to-service communication authenticated via mutual TLS (mTLS) is exempt from standard rate limiting policies.


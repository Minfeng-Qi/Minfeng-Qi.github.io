---
layout: post
title: "Access control for IoT"
date: 2020-09-01 
description: "Using access control to protect IoT privacy"
tag: IoT, Access Control
---

### This article is to record state-of-the-art IoT access control mechanism.

1. **Features of IoT system**

   - **Indoor wayfinding.** Users can benefit from indoor [wayfinding](https://www.jibestream.com/blog/how-the-internet-of-things-is-delivering-experiential-wayfinding) and their accessibility options can be seen on their mobile devices.
   - **Secure access control.** Access credentials are easy to manage and update. Doors can be opened from a distance.
   - **Instant confirmation.** Users can get instant confirmation of access requests.
   - **Convenient interaction.** It provides easy interaction with other users and also provides location details to users.
   - **No physical ID.** Physical ID is not required; therefore, the risk of it being stolen or lost is eliminated.

2. **Challenges**

3. **Featuers of traditional access control model**

   - **High costs** of centralized cloud maintenance and networking equipment. The costs will continue to rise with the proliferation of connected devices.
   - **Low interoperability** due to restricted data exchange with other centralized infrastructures.
   - **Single gateway** is not trustworthy, as it allows gaining access to a whole IoT network by compromising a single device.

4. **Procedures to protect IoT system**

   - Provision devices and systems with unique identities and credentials.
   - Apply authentication and access control mechanisms.
   - Use cryptographic network protocols.
   - Create continuous update and deployment mechanisms.
   - Deploy security auditing and monitoring mechanisms.
   - Build continuous health checks for security mechanisms.
   - Proactively assess the impact of potential security events.
   - Minimize the attack surface of your IoT ecosystem.
   - Avoid unnecessary data access, storage, and transmission.
   - Monitor vulnerability disclosure and threat intelligence sources.

5. **Access control elements**

   | Elements                                                     | Explanation                                         |
   | :----------------------------------------------------------- | --------------------------------------------------- |
   | [**Users**](https://aws.amazon.com/iam/features/manage-users/) | Manage permissions with groups.                     |
   | [**Groups**](https://aws.amazon.com/iam/features/manage-users/) | Manage permissions with groups.                     |
   | [**Permissions**](https://aws.amazon.com/iam/features/manage-permissions/) | Grant least privilege.                              |
   | [**Auditing**](https://aws.amazon.com/cloudtrail/)           | Turn on AWS CloudTrail.                             |
   | [**Password**](https://aws.amazon.com/iam/features/managing-user-credentials/) | Configure a strong password policy.                 |
   | [**MFA**](https://aws.amazon.com/iam/features/mfa/)          | Enable MFA for privileged users.                    |
   | [**Roles**](https://aws.amazon.com/iam/features/manage-roles/) | Use IAM roles for Amazon EC2 instances.             |
   | [**Sharing**](https://aws.amazon.com/identity/federation/)   | Use IAM roles to share access.                      |
   | [**Rotate**](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html) | Rotate security credentials regularly.              |
   | [**Conditions**](http://docs.aws.amazon.com/IAM/latest/UserGuide/PermissionsAndPolicies.html) | Restrict privileged access further with conditions. |

